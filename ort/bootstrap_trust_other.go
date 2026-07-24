//go:build !darwin && !linux

package ort

import "os"

func validateBootstrapPathOwnershipAndMode(string, os.FileInfo) error {
	return nil
}
