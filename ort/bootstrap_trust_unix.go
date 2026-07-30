//go:build darwin || linux

package ort

import (
	"fmt"
	"os"
	"syscall"
)

func validateBootstrapPathOwnershipAndMode(path string, info os.FileInfo) error {
	if info.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("bootstrap path is writable by group or others: %q (mode %04o)", path, info.Mode().Perm())
	}

	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("bootstrap path ownership is unavailable for %q", path)
	}
	if stat.Uid != uint32(os.Geteuid()) { // #nosec G115 -- Unix effective UIDs are non-negative uid_t values; conversion matches syscall.Stat_t.Uid for comparison.
		return fmt.Errorf("bootstrap path %q is owned by uid %d, want current uid %d", path, stat.Uid, os.Geteuid())
	}
	return nil
}
