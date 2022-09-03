# 6D Pose-Estimation

## Real Dataset (Format is similar to LINEMOD and LINEMOD-Occlusion)

![3](https://user-images.githubusercontent.com/61361845/177278919-cd53068f-c53e-4c33-9107-ce7544804f0c.png)


## Synthetic Dataset

LINEMOD.

![3D2](https://user-images.githubusercontent.com/61361845/188285419-a62f572c-be98-4fc0-ab17-9cb841760fd3.png)

![3D3](https://user-images.githubusercontent.com/61361845/188285420-924db41f-633b-4e11-a414-15c7c844e43b.png)

![3D1](https://user-images.githubusercontent.com/61361845/188285422-7eddde5d-c75b-40cd-98f1-d9230d0cef32.png)


Occlusion.

![object_1 2](https://user-images.githubusercontent.com/61361845/177279694-2eb99287-a92f-4833-9b8e-1e70bc0fa802.png)


## To check annotations run:

```
check_annotation.py --object-path ./DatasetLinemod --object-id 1
```

## Testing

Testing is done with the model trained on small real dataset only.

![test-gif](https://user-images.githubusercontent.com/61361845/177279942-9579591f-4e4d-4972-a556-43696f15402b.gif)


