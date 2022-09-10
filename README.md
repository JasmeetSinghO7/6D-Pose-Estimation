# 6D Pose-Estimation

## Real Dataset (Format is similar to LINEMOD and LINEMOD-Occlusion)

![3](https://user-images.githubusercontent.com/61361845/177278919-cd53068f-c53e-4c33-9107-ce7544804f0c.png)


## Synthetic Dataset

LINEMOD.

![11](https://user-images.githubusercontent.com/61361845/189476136-770da231-ec17-49a6-ba77-f81a34af5af4.png)

![6](https://user-images.githubusercontent.com/61361845/189476144-ae55423b-4ac0-4f7d-842f-5b2fa0bcb3e2.png)


Occlusion.

![object_1 2](https://user-images.githubusercontent.com/61361845/177279694-2eb99287-a92f-4833-9b8e-1e70bc0fa802.png)


## To check annotations run:

```
check_annotation.py --object-path ./DatasetLinemod --object-id 1
```

## Testing

Testing is done with the model trained on small real dataset only.

![test-gif](https://user-images.githubusercontent.com/61361845/177279942-9579591f-4e4d-4972-a556-43696f15402b.gif)


