## Results

You can review the results in the output/ directory:

seed_patch_k{<k>}_seed{<seed>}.png — the extracted seed patch

seed_original_k{<k>}_seed{<seed>}.png — the full MNIST image with the seed patch outlined in red

result_k{<k>}_seed{<seed>}.png — the final synthesized image

anim_k{<k>}_seed{<seed>}.gif — an optional GIF animation of the synthesis process


## Usage

Only need to change the patch size for our 289 class purpose, it will generate the results in the output/ directory by default.:

```
python -u mnist_synthesis.py --k 3  --seed 0
python -u mnist_synthesis.py --k 5  --seed 0
python -u mnist_synthesis.py --k 7  --seed 0
python -u mnist_synthesis.py --k 9  --seed 0
python -u mnist_synthesis.py --k 3  --seed 42
python -u mnist_synthesis.py --k 5  --seed 42
python -u mnist_synthesis.py --k 7  --seed 42
python -u mnist_synthesis.py --k 9  --seed 42
```

More Detail:

`python mnist_synthesis.py --win_h 28 --win_w 28 --k 7 --th 0.1 --seed 42 --seed_out output/seed_patch_k7.png --seed_img output seed_original_k7.png --gif output/anim_k7.gif --fps 30 --out output/result_k7.png`


--win_h
Height of the output canvas in pixels. Defaults to 28.

--win_w
Width of the output canvas in pixels. Defaults to 28.

--k
Patch size; each patch is k×k. Defaults to 7.

--th
Normalized error threshold ε for candidate selection. Defaults to 0.1.

--seed
Random seed for reproducible results. Defaults to 42.

--seed_out
(Optional) Path to save the extracted seed patch image.
Defaults to output/seed_patch_k{<k>}_seed{<seed>}.png.

--seed_img
(Optional) Path to save the original 28×28 MNIST image with the seed patch outlined in red.
Defaults to output/seed_original_k{<k>}_seed{<seed>}.png.

--gif
(Optional) Path to save an animation GIF of the synthesis process.
Defaults to output/anim_k{<k>}_seed{<seed>}.gif.

--fps
Frame rate (frames per second) for the output GIF. Defaults to 30.

--out
(Optional) Path to save the final synthesized image.
Defaults to output/result_k{<k>}_seed{<seed>}.png.

