# nestedloopsfusion
LLVM branching optimization transformation pass for GPUs

To compile the Loops Fusion transformation pass, you have to get a working and up-to date version of LLVM/Clang.

**Build LLVM**

Assume you compiled Clang to be installed as a local user, using CMake config like this (instructions valid for LLVM 9.0):
```bash
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=debug -DLLVM_ENABLE_PROJECTS="clang;llvm;clang-tools-extra;compiler-rt" -DCMAKE_INSTALL_PREFIX=/home/username/local -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;WebAssembly" ../llvm-project/llvm/
```


**Build LoopF transformation pass module**

After that, the pass can be compiled as follows:
```bash
mkdir build
cd build
env CC=clang -CMAKE_PREFIX_PATH=/home/username/local -DCMAKE_INSTALL_PREFIX_PATH=/home/username/local ../
make
```
After that, LoopF llvm pass will become available as a plugin module for `opt` utility.

**Build cudatest**

Assuming you already have CUDA installed, `cudatest` benchmark can be compiled like this:

```bash
clang++ cudatest.cu  -L/usr/local/cuda/lib64/ -I/usr/local/cuda/samples/common/inc -lcudart_static -ldl -lrt -pthread    -o cudatest_ref --cuda-gpu-arch=sm_70
```
To test it out, run `cudatest_ref 1 1234 3000 31 2`
It should print something like:
```
 ...
 356379
 Time Sum Avg Avgt/elem 1.949342 15047055 470220 241219.915248

 ```
 
 **Applying transformation pass to the benchmark**
 To build the transformed version of `cudatest`, you first have to create a build script that is based on Clang's compilation process.
 To do that, we can run the compilation command with `-###` option that tells `clang` to just print the compilation commands it is about to run, instead of running them.
 
 ```bash
 clang++ cudatest.cu  -L/usr/local/cuda/lib64/ -I/usr/local/cuda/samples/common/inc -lcudart_static -ldl -lrt -pthread    -o cudatest_transformed --cuda-gpu-arch=sm_70 -O3 -save-temps -### 2> ./make_transformed.sh
 ```
 Now open this `make_transformed.sh`. It will look like this:
 ```bash
clang version 9.0.0 (https://github.com/llvm/llvm-project.git 635b988578505eee09ff304974bc2a72becb66d3)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/username/local/bin
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/usr/local/cuda-10.1/bin/ptxas" "-m64" ..
 "/usr/local/cuda-10.1/bin/fatbinary" "--cuda" ..
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/home/username/local/bin/clang-9" "-cc1" ..
 "/home/username/local/bin/clang-9" "-cc1as" ..
 "/usr/bin/ld" "-z" "relro" "--hash-style=gnu" ..
 ```
 
 Now you have to remove `"` symbols from this file and make it a `bash` script, something like this:
  ```bash
#!/bin/bash
 /home/username/local/bin/clang-9 -cc1 .. 
 /home/username/local/bin/clang-9 -cc1 .. 
 /home/username/local/bin/clang-9 -cc1 .. 
 /usr/local/cuda-10.1/bin/ptxas -m64 ..
 /usr/local/cuda-10.1/bin/fatbinary --cuda ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1as ..
 /usr/bin/ld -z relro --hash-style=gnu ..
 ```
 Make it executable `chmod +x make_transformed.sh` and run it to test if compilation script works.
 
 Now, the transformation command should be injected into this script after the *second* line, like this:
 ```bash
 #!/bin/bash
 /home/username/local/bin/clang-9 -cc1 .. 
 /home/username/local/bin/clang-9 -cc1 ..  -o cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc ..
 opt -load ../LoopF/build/LoopF/libLoopF.so -simplifycfg -loop-rotate -loopf cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc |opt -O3 > cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc_mod
 mv cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc_orig
 mv cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc_mod cudatest-cuda-nvptx64-nvidia-cuda-sm_70.bc
 /home/username/local/bin/clang-9 -cc1 .. 
 /usr/local/cuda-10.1/bin/ptxas -m64 ..
 /usr/local/cuda-10.1/bin/fatbinary --cuda ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1 ..
 /home/username/local/bin/clang-9 -cc1as ..
 /usr/bin/ld -z relro --hash-style=gnu ..
 ```
 After that, you can run `make_transformed.sh` to produce `cudatest_transformed` which is the same program as `cudatest_ref`, with its GPU kernel code transformed with Nested Loops Fusion transformation pass.
 You can run it with the same parameters as original `cudatest_transformed 1 1234 3000 31 2`, but now it will finish much faster and show better benchmark values:
 
```
...
  356379
 Time Sum Avg Avgt/elem 0.111188 15047055 470220 4229054.838467
```
 
 
 
 
 
 
 
 
