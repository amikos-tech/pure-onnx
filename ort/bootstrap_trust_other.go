//go:build !darwin && !linux

package ort

import (
	"os"
	"runtime"
)

func validateBootstrapPathOwnershipAndMode(
	_ string,
	_ os.FileInfo,
	_ bool,
) error {
	switch runtime.GOOS {
	case "windows":
		// POSIX mode bits and Unix UIDs do not describe Windows ACL trust.
		// Platform-neutral type, symlink, manifest, and hash checks still apply.
		return nil
	default:
		// Other non-Unix targets have no portable ownership/mode contract here.
		// Platform-neutral cache integrity checks remain mandatory.
		return nil
	}
}
