#!/bin/bash

#SBATCH -p barbun-cuda      							# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ckatar          							# Kullanici adi
#SBATCH -J train_att_unet_dice_bce_2016							# Gonderilen isin ismi
#SBATCH -o train_att_unet_dice_bce2016.out			  			        	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        							# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                							# Gorev kac node'da calisacak?
#SBATCH -n 1                							# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20							# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=24:00:00    							# Sure siniri koyun. (HH:MM:SS)
#SBATCH --mail-type=begin        						# send email when job begins
#SBATCH --mail-type=end          						# send email when job ends
#SBATCH --mail-user=cihan.katar@tau.edu.tr

eval "$(/truba/sw/centos7.3/lib/anaconda3/2021.11/bin/conda shell.bash hook)"

conda activate segmentation
cd /truba/home/ckatar/Desktop/Att-Next-TopoLoss
python train_att_unet_dice_bce_2016.py -p=False -i=256 -b=8 -x=0.5 -o=0.5 -e=200