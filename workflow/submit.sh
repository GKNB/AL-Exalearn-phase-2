#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug
#PBS -A RECUP

source /home/twang3/useful_script/conda_exalearn.sh
export MPICH_GPU_SUPPORT_ENABLED=1

seed=40
work_dir="/lus/eagle/projects/RECUP/twang/exalearn_stage2/"
exe_dir="${work_dir}/executable/"
exp_dir="${work_dir}/experiment/seed_${seed}/"
data_dir="${work_dir}/data/seed_${seed}/"

echo "Logging: Start! seed = ${seed}"
echo "Logging: data_dir = ${data_dir}"
echo "Logging: Doing cleaning"
mv ${exp_dir} /tmp/
mv ${data_dir} /tmp/

mkdir -p ${exp_dir}
cd ${exp_dir}

################################  Start real job  ##############################

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python3 ${work_dir}/prepare_data_dir.py --seed ${seed}

nthread=32

echo "Logging: Start base simulation and merge!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
    python ${exe_dir}/simulation_sample.py \
            6000 ${seed} \
            ${data_dir}/base/config/config_1001460_cubic.txt \
            ${data_dir}/base/config/config_1522004_trigonal.txt \
            ${data_dir}/base/config/config_1531431_tetragonal.txt

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data tetragonal ${nthread}
echo "Logging: End base simulation and merge"

echo "Logging: Start test simulation and merge!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
     python ${exe_dir}/simulation_sample.py \
            2000 $((${seed} + 1)) \
            ${data_dir}/test/config/config_1001460_cubic.txt \
            ${data_dir}/test/config/config_1522004_trigonal.txt \
            ${data_dir}/test/config/config_1531431_tetragonal.txt

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data tetragonal ${nthread}
echo "Logging: End test simulation and merge"

nthread=10

echo "Logging: Start study simulation and merge!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
     python ${exe_dir}/simulation_sweep.py \
            2000 \
            ${data_dir}/study/config/config_1001460_cubic.txt \
            ${data_dir}/study/config/config_1522004_trigonal.txt \
            ${data_dir}/study/config/config_1531431_tetragonal.txt

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data tetragonal ${nthread}
echo "Logging: End study simulation and merge!"

echo "Logging: Start training, phase 0"
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
    python ${exe_dir}/train_step1.py --device=gpu --data_dir=${data_dir}
echo "Logging: End training phase 0"

nthread=32

echo "Logging: Start resample simulation and merge, phase 1!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
     python ${exe_dir}/simulation_resample.py \
            $((${seed} + 2)) \
            ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
            ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
            ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
            ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
            ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
            ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread}
echo "Logging: End resample simulation and merge, phase 1!"

rm AL-freq.npy

echo "Logging: Start training, phase 1"
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
    python ${exe_dir}/train_step2.py --device=gpu --phase_idx=1 --data_dir=${data_dir}
echo "Logging: End training, phase 1"

echo "Logging: Start resample simulation and merge, phase 2!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
    python ${exe_dir}/simulation_resample.py \
            $((${seed} + 3)) \
            ${data_dir}/AL_phase_2/config/config_1001460_cubic.txt \
            ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
            ${data_dir}/AL_phase_2/config/config_1522004_trigonal.txt \
            ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
            ${data_dir}/AL_phase_2/config/config_1531431_tetragonal.txt \
            ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data tetragonal ${nthread}
echo "Logging: End resample simulation and merge, phase 2!"

rm AL-freq.npy

echo "Logging: Start training, phase 2"
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
    python ${exe_dir}/train_step2.py --device=gpu --phase_idx=2 --data_dir=${data_dir}
echo "Logging: End training, phase 2"

echo "Logging: Start resample simulation and merge, phase 3!"
mpiexec -n ${nthread} --ppn ${nthread} --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
     python ${exe_dir}/simulation_resample.py \
            $((${seed} + 4)) \
            ${data_dir}/AL_phase_3/config/config_1001460_cubic.txt \
            ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
            ${data_dir}/AL_phase_3/config/config_1522004_trigonal.txt \
            ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
            ${data_dir}/AL_phase_3/config/config_1531431_tetragonal.txt \
            ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data cubic ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data trigonal ${nthread}
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data tetragonal ${nthread}
echo "Logging: End resample simulation and merge, phase 3!"

rm AL-freq.npy

echo "Logging: Start training, phase 3"
mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
    python ${exe_dir}/train_step2.py --device=gpu --phase_idx=3 --data_dir=${data_dir}
echo "Logging: End training, phase 3"

echo "Logging: All done for seed = ${seed}"
