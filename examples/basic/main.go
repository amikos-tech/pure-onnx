package main

import (
	"fmt"
	"log"

	"github.com/amikos-tech/onnx-purego/ort"
)

func main() {
	// Initialize ONNX Runtime environment
	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatal("Failed to initialize ONNX Runtime:", err)
	}
	defer ort.DestroyEnvironment()

	fmt.Println("ONNX Runtime initialized successfully")
	fmt.Printf("Version: %s\n", ort.GetVersionString())

	// Example: Create and run a session (to be implemented)
	// session, err := ort.NewSession("model.onnx", nil)
	// if err != nil {
	//     log.Fatal("Failed to create session:", err)
	// }
	// defer session.Destroy()
}