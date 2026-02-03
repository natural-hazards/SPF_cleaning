# SPF_cleaning
Cleaning scripts for SPF

```
python find_errors.py   --mseed_list mseeds.txt   --ok_mseed_list ok_mseeds.txt   --output output.csv   --n_fft 4096   --seed 0   --ae_epochs 60   --ae_percentile 99   --ae_device cuda   --ae_model_out ae_model.pt
python visualize.py --csv output.csv --out output.pdf --max-pages 1000
```
