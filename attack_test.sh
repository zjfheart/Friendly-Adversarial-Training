python attack_test.py --net 'WRN_madry' --depth 32 --model_path './FAT_models/fat_wrn32-10_eps0.031.pth.tar' --method 'dat'
python attack_test.py --net 'WRN_madry' --depth 32 --model_path './FAT_models/fat_wrn32-10_eps0.062.pth.tar' --method 'dat'

python attack_test.py --net 'WRN' --depth 34 --model_path './FAT_models/fat_for_trades_wrn34-10_eps0.031_beta1.0.pth.tar' --method 'trades'
python attack_test.py --net 'WRN' --depth 34 --model_path './FAT_models/fat_for_trades_wrn34-10_eps0.031_beta6.0.pth.tar' --method 'trades'
python attack_test.py --net 'WRN' --depth 34 --model_path './FAT_models/fat_for_trades_wrn34-10_eps0.062_beta6.0.pth.tar' --method 'trades'

python attack_test.py --net 'WRN' --depth 58 --model_path './FAT_models/fat_for_trades_wrn58-10_eps0.031_beta6.0.pth.tar' --method 'trades'
python attack_test.py --net 'WRN' --depth 58 --model_path './FAT_models/fat_for_trades_wrn58-10_eps0.062_beta6.0.pth.tar' --method 'trades'
