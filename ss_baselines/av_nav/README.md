# Audio-Visual Navigation (AV-Nav) Model

## Details
This folder provides the code of the model as well as the training/evaluation configurations used in the 
[SoundSpaces: Audio-Visual Navigation in 3D Environments](https://arxiv.org/pdf/1912.11474.pdf) paper.
Use of this model is the same as described in the usage section of the main README file.
Pretrained weights are provided.

## Evaluating pretrained model
```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/audionav/av_nav/replica/heard.pth 
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml EVAL_CKPT_PATH_DIR data/pretrained_weights/audionav/av_nav/replica/unheard.pth EVAL.SPLIT test_multiple_unheard 
```


## Citation
If you use this model in your research, please cite the following paper:
```
@inproceedings{chen20soundspaces,
  title     =     {SoundSpaces: Audio-Visual Navigaton in 3D Environments,
  author    =     {Changan Chen and Unnat Jain and Carl Schissler and Sebastia Vicenc Amengual Gari and Ziad Al-Halah and Vamsi Krishna Ithapu and Philip Robinson and Kristen Grauman},
  booktitle =     {ECCV},
  year      =     {2020}
}
```
 python ss_baselines/av_nav/run.py --run-type eval --exp-config /home/Disk/sound-space/ss_baselines/av_nav/config/audionav/mp3d/val_telephone/audiogoal_depth.yaml --model-dir data/models/mp3d/audiogoal_depth EVAL_CKPT_PATH_DIR /home/Disk/sound-space/data/pretrained_weights/audionav/av_nav/mp3d/heard.pth VIDEO_OPTION [\"disk\"] TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False TASK_CONFIG.TASK.SENSORS [\"POINTGOAL_WITH_GPS_COMPASS_SENSOR\",\"SPECTROGRAM_SENSOR\",\"AUDIOGOAL_SENSOR\"] SENSORS [\"RGB_SENSOR\",\"DEPTH_SENSOR\"] EXTRA_RGB True TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE False DISPLAY_RESOLUTION 512 TEST_EPISODE_COUNT 2
