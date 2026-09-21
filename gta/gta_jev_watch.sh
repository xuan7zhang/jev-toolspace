#!/bin/bash
# Start run_gta_jev.sh once a non-reserved node (not kn068) has a GPU with >=31 GB free (tool server) and another with >=27 GB free (7B server).
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
J=/datasets/omni_pretraining/gta2/results/jev; RES=/datasets/omni_pretraining/bfcl/runs/overnight/reserved.txt
while true; do
  for line in $(squeue -u xzhan576 -h -t R -o "%i:%N"); do
    jb=${line%%:*}; nd=${line##*:}; [ "$nd" = kn068 ] && continue; grep -qx "$nd" $RES 2>/dev/null && continue
    free=$(timeout 40 srun --jobid=$jb --overlap --ntasks=1 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits </dev/null 2>/dev/null)
    tg=$(echo "$free" | awk -F', ' '$2>=31000{print $1; exit}'); [ -z "$tg" ] && continue
    lg=$(echo "$free" | awk -F', ' -v t=$tg '$1!=t && $2>=27000{print $1; exit}'); [ -z "$lg" ] && continue
    echo "$(date) launching on $nd job=$jb LG=$lg TG=$tg"; LG=$lg TG=$tg JB=$jb bash $J/run_gta_jev.sh > $J/run_gta_jev.out 2>&1; echo "$(date) run_gta_jev finished rc=$?"; exit 0
  done
  sleep 300
done
