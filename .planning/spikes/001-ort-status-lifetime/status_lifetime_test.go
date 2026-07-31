package ortstatuslifetime

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"testing"

	ort "github.com/amikos-tech/pure-onnx/ort"
)

// statusError is the smallest useful prototype of the planned ort.ORTError.
// It deliberately stores only Go-owned values.
type statusError struct {
	Op      string
	Code    ort.ErrorCode
	Message string
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s: ORT code %d: %s", e.Op, e.Code, e.Message)
}

type statusOps struct {
	getCode     func(uintptr) ort.ErrorCode
	copyMessage func(uintptr) string
	release     func(uintptr)
}

// statusToErrorPrototype snapshots every status-owned value before releasing
// the native status. The defer is installed before any status accessor runs so
// future helper changes cannot accidentally bypass ReleaseStatus.
func statusToErrorPrototype(status uintptr, op string, ops statusOps) error {
	if status == 0 {
		return nil
	}
	defer ops.release(status)

	return &statusError{
		Op:      op,
		Code:    ops.getCode(status),
		Message: ops.copyMessage(status),
	}
}

type fakeStatus struct {
	code    ort.ErrorCode
	message []byte
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

func (s *fakeStatusStore) create(code ort.ErrorCode, message string) uintptr {
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

func (s *fakeStatusStore) code(handle uintptr) ort.ErrorCode {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.statuses[handle].code
}

func (s *fakeStatusStore) copyMessage(handle uintptr) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	message := s.statuses[handle].message
	return string(message[:len(message)-1])
}

func (s *fakeStatusStore) release(handle uintptr) {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := s.statuses[handle]
	for i := range status.message {
		status.message[i] = 'x'
	}
	s.releaseCount[handle]++
}

func (s *fakeStatusStore) releases(handle uintptr) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.releaseCount[handle]
}

func (s *fakeStatusStore) ops() statusOps {
	return statusOps{
		getCode:     s.code,
		copyMessage: s.copyMessage,
		release:     s.release,
	}
}

func TestStatusSnapshotSurvivesRelease(t *testing.T) {
	store := newFakeStatusStore()
	handle := store.create(ort.ErrorCodeInvalidArgument, "shape mismatch")

	err := statusToErrorPrototype(handle, "run inference", store.ops())

	var got *statusError
	if !errors.As(err, &got) {
		t.Fatalf("errors.As did not find *statusError in %T", err)
	}
	if got.Op != "run inference" {
		t.Fatalf("operation mismatch: got %q", got.Op)
	}
	if got.Code != ort.ErrorCodeInvalidArgument {
		t.Fatalf("code mismatch: got %d", got.Code)
	}
	if got.Message != "shape mismatch" {
		t.Fatalf("message changed after release: got %q", got.Message)
	}
	if releases := store.releases(handle); releases != 1 {
		t.Fatalf("release count: got %d, want 1", releases)
	}

	runtime.KeepAlive(store)
}

func TestZeroStatusIsSuccessAndIsNotReleased(t *testing.T) {
	store := newFakeStatusStore()
	if err := statusToErrorPrototype(0, "noop", store.ops()); err != nil {
		t.Fatalf("zero status returned an error: %v", err)
	}
	if releases := store.releases(0); releases != 0 {
		t.Fatalf("zero status release count: got %d, want 0", releases)
	}
}

func TestConcurrentStatusSnapshotsReleaseExactlyOnce(t *testing.T) {
	const workers = 256

	store := newFakeStatusStore()
	handles := make([]uintptr, workers)
	for i := range handles {
		handles[i] = store.create(ort.ErrorCodeRuntimeException, fmt.Sprintf("failure-%03d", i))
	}

	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i, handle := range handles {
		wg.Add(1)
		go func(index int, status uintptr) {
			defer wg.Done()

			err := statusToErrorPrototype(status, "concurrent call", store.ops())
			var got *statusError
			if !errors.As(err, &got) {
				errs <- fmt.Errorf("worker %d: errors.As failed", index)
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
		if releases := store.releases(handle); releases != 1 {
			t.Errorf("status %d release count: got %d, want 1", handle, releases)
		}
	}

	runtime.KeepAlive(store)
}
