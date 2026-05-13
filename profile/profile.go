package profile

import (
	"flag"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
var memprofile = flag.String("memprofile", "", "write memory profile to `file`")

func CPU() func() {
	flag.Parse()

	if *cpuprofile == "" {
		return nil
	}

	var file *os.File

	if f, err := os.Create(*cpuprofile); err != nil {
		log.Fatal("could not create CPU profile: ", err)
	} else {
		file = f
	}

	if err := pprof.StartCPUProfile(file); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	return func() {
		pprof.StopCPUProfile()

		if err := file.Close(); err != nil {
			log.Fatal(err)
		}
	}
}

func Mem() {
	flag.Parse()

	if *memprofile == "" {
		return
	}

	var file *os.File

	if f, err := os.Create(*memprofile); err != nil {
		log.Fatal("could not create memory profile: ", err)
	} else {
		file = f

		defer file.Close()
	}

	runtime.GC()

	if err := pprof.Lookup("allocs").WriteTo(file, 0); err != nil {
		log.Fatal("could not write memory profile: ", err)
	}
}
