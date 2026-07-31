//go:build darwin || linux

package ort

import (
	"fmt"
	"os"
	"syscall"
)

func validateBootstrapPathOwnershipAndMode(
	path string,
	info os.FileInfo,
	allowSharedCache bool,
) error {
	permissions := info.Mode().Perm()
	if permissions&0o002 != 0 {
		return fmt.Errorf("bootstrap path is writable by others: %q (mode %04o)", path, permissions)
	}
	if !allowSharedCache && permissions&0o020 != 0 {
		return fmt.Errorf("bootstrap path is writable by group: %q (mode %04o)", path, permissions)
	}

	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("bootstrap path ownership is unavailable for %q", path)
	}
	if allowSharedCache {
		return nil
	}
	effectiveUID := uint32(os.Geteuid()) // #nosec G115 -- Unix effective UIDs are non-negative uid_t values; conversion matches syscall.Stat_t.Uid for comparison.
	if stat.Uid != effectiveUID {
		return fmt.Errorf("bootstrap path %q is owned by uid %d, want current uid %d", path, stat.Uid, os.Geteuid())
	}
	return nil
}
