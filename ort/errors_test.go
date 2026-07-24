package ort

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type fakeStatus struct {
	code    ErrorCode
	message []byte
	events  []string
}

type fakeStatusStore struct {
	mu           sync.Mutex
	next         uintptr
	statuses     map[uintptr]*fakeStatus
	releaseCount map[uintptr]int
}

func newFakeStatusStore() *fakeStatusStore {
	return &fakeStatusStore{
		statuses:     make(map[uintptr]*fakeStatus),
		releaseCount: make(map[uintptr]int),
	}
}

func (s *fakeStatusStore) create(code ErrorCode, message string) uintptr {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.next++
	handle := s.next
	s.statuses[handle] = &fakeStatus{
		code:    code,
		message: append([]byte(message), 0),
	}
	return handle
}

func (s *fakeStatusStore) code(handle uintptr) ErrorCode {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	status.events = append(status.events, "code")
	return status.code
}

func (s *fakeStatusStore) copyMessage(handle uintptr) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	status.events = append(status.events, "message")
	return string(status.message[:len(status.message)-1])
}

func (s *fakeStatusStore) release(handle uintptr) {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	status.events = append(status.events, "release")
	for i := range status.message {
		status.message[i] = 'x'
	}
	s.releaseCount[handle]++
}

func (s *fakeStatusStore) snapshot(handle uintptr) (message string, releases int, events []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	return string(status.message), s.releaseCount[handle], append([]string(nil), status.events...)
}

func (s *fakeStatusStore) ops() statusOps {
	return statusOps{
		getCode:     s.code,
		copyMessage: s.copyMessage,
		release:     s.release,
	}
}

func TestStatusToError(t *testing.T) {
	t.Run("zero status is a no-op", func(t *testing.T) {
		var calls atomic.Int32
		ops := statusOps{
			getCode: func(uintptr) ErrorCode {
				calls.Add(1)
				return ErrorCodeFail
			},
			copyMessage: func(uintptr) string {
				calls.Add(1)
				return "unexpected"
			},
			release: func(uintptr) {
				calls.Add(1)
			},
		}

		if err := statusToErrorWithOps(0, "no operation", ops); err != nil {
			t.Fatalf("zero status returned an error: %v", err)
		}
		if got := calls.Load(); got != 0 {
			t.Fatalf("zero status invoked %d callbacks, want 0", got)
		}
	})

	t.Run("copies fields before releasing exactly once", func(t *testing.T) {
		store := newFakeStatusStore()
		handle := store.create(ErrorCodeInvalidArgument, "shape mismatch")

		err := statusToErrorWithOps(handle, "run inference", store.ops())

		var got *ORTError
		if !errors.As(err, &got) {
			t.Fatalf("errors.As did not find *ORTError in %T", err)
		}
		if got.Operation != "run inference" {
			t.Fatalf("operation = %q, want %q", got.Operation, "run inference")
		}
		if got.Code != ErrorCodeInvalidArgument {
			t.Fatalf("code = %d, want %d", got.Code, ErrorCodeInvalidArgument)
		}
		if got.Message != "shape mismatch" {
			t.Fatalf("message = %q, want %q", got.Message, "shape mismatch")
		}

		nativeMessage, releases, events := store.snapshot(handle)
		if nativeMessage == "shape mismatch\x00" {
			t.Fatal("release did not overwrite the fake native message")
		}
		if got.Message != "shape mismatch" {
			t.Fatalf("Go-owned message changed after release: %q", got.Message)
		}
		if releases != 1 {
			t.Fatalf("release count = %d, want 1", releases)
		}
		wantEvents := []string{"code", "message", "release"}
		if fmt.Sprint(events) != fmt.Sprint(wantEvents) {
			t.Fatalf("callback order = %v, want %v", events, wantEvents)
		}
	})

	t.Run("releases when an accessor panics", func(t *testing.T) {
		tests := []struct {
			name string
			ops  func(*atomic.Int32) statusOps
		}{
			{
				name: "code accessor",
				ops: func(releases *atomic.Int32) statusOps {
					return statusOps{
						getCode: func(uintptr) ErrorCode {
							panic("code accessor failed")
						},
						copyMessage: func(uintptr) string {
							return "unreachable"
						},
						release: func(uintptr) {
							releases.Add(1)
						},
					}
				},
			},
			{
				name: "message accessor",
				ops: func(releases *atomic.Int32) statusOps {
					return statusOps{
						getCode: func(uintptr) ErrorCode {
							return ErrorCodeFail
						},
						copyMessage: func(uintptr) string {
							panic("message accessor failed")
						},
						release: func(uintptr) {
							releases.Add(1)
						},
					}
				},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var releases atomic.Int32
				func() {
					defer func() {
						if recover() == nil {
							t.Fatal("accessor did not panic")
						}
					}()
					_ = statusToErrorWithOps(1, "panic test", tc.ops(&releases))
				}()
				if got := releases.Load(); got != 1 {
					t.Fatalf("release count = %d, want 1", got)
				}
			})
		}
	})

	t.Run("concurrent conversions retain independent messages", func(t *testing.T) {
		const workers = 256

		store := newFakeStatusStore()
		handles := make([]uintptr, workers)
		for i := range handles {
			handles[i] = store.create(ErrorCodeRuntimeException, fmt.Sprintf("failure-%03d", i))
		}

		var wg sync.WaitGroup
		errs := make(chan error, workers)
		for i, handle := range handles {
			wg.Add(1)
			go func(index int, status uintptr) {
				defer wg.Done()

				err := statusToErrorWithOps(status, "concurrent call", store.ops())
				var got *ORTError
				if !errors.As(err, &got) {
					errs <- fmt.Errorf("worker %d: errors.As failed for %T", index, err)
					return
				}
				wantMessage := fmt.Sprintf("failure-%03d", index)
				if got.Message != wantMessage {
					errs <- fmt.Errorf("worker %d: message = %q, want %q", index, got.Message, wantMessage)
				}
			}(i, handle)
		}
		wg.Wait()
		close(errs)

		for err := range errs {
			t.Error(err)
		}
		for _, handle := range handles {
			_, releases, _ := store.snapshot(handle)
			if releases != 1 {
				t.Errorf("status %d release count = %d, want 1", handle, releases)
			}
		}
	})
}

func TestORTError(t *testing.T) {
	err := &ORTError{
		Operation: "load model",
		Code:      ErrorCodeInvalidGraph,
		Message:   "invalid node",
	}

	for _, want := range []string{"load model", fmt.Sprint(ErrorCodeInvalidGraph), "invalid node"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Error() = %q, want it to contain %q", err.Error(), want)
		}
	}
}

func TestErrorSentinel(t *testing.T) {
	sentinels := []struct {
		name string
		err  error
	}{
		{name: "invalid argument", err: ErrInvalidArgument},
		{name: "not initialized", err: ErrNotInitialized},
		{name: "destroyed", err: ErrDestroyed},
		{name: "shared library not found", err: ErrSharedLibraryNotFound},
		{name: "unsupported platform", err: ErrUnsupportedPlatform},
	}

	for _, tc := range sentinels {
		t.Run(tc.name, func(t *testing.T) {
			wrapped := fmt.Errorf("outer context: %w", fmt.Errorf("operation context: %w", tc.err))
			if !errors.Is(wrapped, tc.err) {
				t.Fatalf("errors.Is(%v, %v) = false, want true", wrapped, tc.err)
			}

			nativeErr := &ORTError{
				Operation: "native operation",
				Code:      ErrorCodeInvalidArgument,
				Message:   "native failure",
			}
			if errors.Is(nativeErr, tc.err) {
				t.Fatalf("native *ORTError unexpectedly matches %v", tc.err)
			}
		})
	}
}
