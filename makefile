NVCC=nvcc
# Add -DDEBUG to verify correctness (prints initial and final system state).
# Remove it for timing runs so I/O does not dominate the measurement.
FLAGS=-O2 -DDEBUG
LIBS=
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $<
compute.o: compute.cu config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $<
clean:
	rm -f *.o nbody
