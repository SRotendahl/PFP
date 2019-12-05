SIZES = 100 1000 10000 100000 1000000 5000000

three_%_f32s:
	futhark dataset -b --f32-bounds=-10000.0:10000.0 -g [$*][10][10]f32 > $@

two_%_i32s:
	futhark dataset -b --i32-bounds=-10000:10000 -g [$*]i32 -g [$*]i32 > $@

one_%_i32s:
	futhark dataset -b --i32-bounds=-10000:10000 -g [$*]i32 > $@

exercise4_%_data:
	futhark dataset --i32-bounds=10:1000 -g 100f32 -g 0.5f32 \
				-g $*i32 -g $*i32 -g 10i32 > $@

exercise_1_1-opencl.json: $(SIZES:%=two_%_i32s) exercise_1_1.fut
	futhark bench --backend=opencl \
			--json=exercise_1_1-opencl.json \
			exercise_1_1.fut

exercise_1_1-c.json: $(SIZES:%=two_%_i32s) exercise_1_1.fut
	futhark bench --backend=c \
			--json=exercise_1_1-c.json \
			exercise_1_1.fut

BIF-scan-opencl.json: $(SIZES:%=one_%_i32s) BIFscan.fut
	futhark bench --backend=opencl --json BIFscan-opencl.json BIFscan.fut

scan_hillis_steele-opencl.json: $(SIZES:%=one_%_i32s) scan_hillis_steele.fut
	futhark bench --backend=opencl \
			--json scan_hillis_steele-opencl.json \
			scan_hillis_steele.fut

scan_work_efficient-opencl.json: $(SIZES:%=one_%_i32s) scan_work_efficient.fut
	futhark bench --backend=opencl \
			--json scan_work_efficient-opencl.json \
			scan_work_efficient.fut

ising-c.json: $(SIZES:%=exercise4_%_data) ising-handout/ising.fut
	futhark bench --backend=c \
			--json ising-c.json ising-handout/ising.fut

ising-opencl.json: $(SIZES:%=exercise4_%_data) ising-handout/ising.fut
	futhark bench --backend=opencl \
			--json ising-opencl.json ising-handout/ising.fut

task4-data: $(SIZES:%=three_%_f32s) matrix-inversion.fut

task4-no-tune.json:
	futhark bench --backend=opencl \
			--json=matrix-inversion-no-tune.json --no-tuning \
			matrix-inversion.fut

task4-tune.json:
	FUTHARK_INCREMENTAL_FLATTENING=1 futhark autotune --backend=opencl matrix-inversion.fut
			FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --json=matrix-inversion-tune.json matrix-inversion.fut
