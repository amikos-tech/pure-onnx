package main

import (
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/amikos-tech/pure-onnx/ort"
)

func getDefaultLibraryPath() string {
	switch runtime.GOOS {
	case "darwin":
		return "/usr/local/lib/libonnxruntime.dylib"
	case "linux":
		return "/usr/lib/libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	default:
		return ""
	}
}

func main() {
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		libPath = getDefaultLibraryPath()
	}

	ort.SetSharedLibraryPath(libPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatal("Failed to initialize ONNX Runtime:", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("Failed to destroy environment: %v", err)
		}
	}()

	fmt.Println("ONNX Runtime initialized successfully")
	fmt.Printf("Version: %s\n", ort.GetVersionString())
	fmt.Printf("Is initialized: %v\n", ort.IsInitialized())

	// Example: Create and run a session (to be implemented)
	// session, err := ort.NewSession("model.onnx", nil)
	// if err != nil {
	//     log.Fatal("Failed to create session:", err)
	// }
	// defer session.Destroy()
}
