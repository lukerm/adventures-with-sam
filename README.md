# Let it Segment: A Gift from SAM

With the release of the Segment Anything Model[^1] (SAM) released by Meta AI Research last year, the lie of the land changed
quite substantially in Computer Vision, as now images could be segmented easily, with great results even zero-shot. With 
the release of SAM2[^2] earlier this year, I wanted to get hands on and experiment with these models myself. 

[^1]: Kirillov, A., Mintun, E. et al. (2023). [Segment Anything](https://arxiv.org/abs/2304.02643) _arXiv preprint_.

[^2]: Ravi, N., Gabeur, V. et al. (2024). [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) 
_arXiv preprint_.

This post walks you through how SAM2 could be used in practice, provides a mini analysis of segmentation results and will
be released with code so that you can explore further if you want to. This could be expanded to interesting use cases, 
such as facilitating object grasping in robotic systems, branded product addition or removal in marketing images, or
mapping changes in forested areas from satellite imagery over time for environmental monitoring.

If you want to jump straight to the Experiments & Analysis section, click [here](#experiments-and-analysis).

### Methodology

This work naturally has the [SAM2](https://github.com/facebookresearch/sam2) project as a core dependency, and that's
where I started to build from. The [`setup_cpu.sh`](https://github.com/lukerm/adventures-with-sam/blob/main/sam_src/bin/setup_cpu.sh)
script documents the installation steps that I went through in order to install PyTorch and other dependencies, and then
downloading the model checkpoints. I found that the CPU on my local laptop was sufficient for experimentation on a handful 
of images, so long as I [down-sampled](https://github.com/lukerm/adventures-with-sam/blob/main/sam_src/bin/prepare_image.sh) 
them to a smaller size first. 

Although I don't want to talk at length about the underlying papers, one interesting contrast between SAM and SAM2 that
jumped out at me is as follows: the former uses a Vision Transformer[^3] (ViT) for its image encoder, whereas the latter 
uses the more recent Hiera ("Hierarchical Vision Transformer") models[^4].

[^3]: Dosovitskiy, A., Beyer, L. et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) _arXiv preprint_.

[^4]: Ryali, C., Hu, Y-T. et al. (2020). [Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/abs/2306.00989) _arXiv preprint_.


As SAM2 is designed to work with videos (SAM is image-only), it has introduced a sophisticated memory mechanism, where 
the memory attention block attends to the "memory bank", which is a store of past frames. This won't be relevant for us,
however, as we are working with images (or single-frame videos).

SAM2 comes in a variety of sizes (tiny, small, etc.), and I chose to compare the "large" and "tiny" versions in my experiments, 
which respectively contain 224 million (40M) parameters baked into a checkpoint file of size 857MB (149MB). 

Since we are performing generic segmentation in this study (rather than using point prompts), we make use of the 
`SAM2AutomaticMaskGenerator` class to generate predicted masks. My [`run_sam2.py`](https://github.com/lukerm/adventures-with-sam/blob/main/sam_src/run_sam2.py)
script then utilizes the masks to create multiple colourful overlays on the original image. You can pass in a whole image
directory as an argument if you want to run it over several images. After the setup, it can be run as in this example:

```
$ cd $HOME/adventures-with-sam
$ PYTHONPATH=. python -m sam_src.run_sam2 --img-dir $HOME/adventures-with-sam/data/img/xmas/small --model-type large --save-segment-imgs 
Processing 0 / 5 images (IMG_20241218_152901.small.jpg)
/home/luke/adventures-with-sam/sam_src/run_sam2.py:105: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
  col_map = plt.cm.get_cmap('hsv', len(masks))
Processed image in 96.192368 seconds
Processing 1 / 5 images (IMG_20241218_153000.small.jpg)
Processed image in 111.612193 seconds
Processing 2 / 5 images (IMG_20241219_085200.small.jpg)
Processed image in 111.478998 seconds
Processing 3 / 5 images (IMG_20241219_141043.small.jpg)
Processed image in 112.866955 seconds
Processing 4 / 5 images (IMG_20241219_141107.small.jpg)
Processed image in 114.209705 seconds
```

You can see that each image takes between 90-120 seconds to process, which could be dramatically sped up by using a GPU.

Note: the `--save-segment-imgs` flag optionally saves each individual mask to a separate images saved in subdirectories.

### Experiments and Analysis

Let's start with a picture of a pair of cute snowman decorations. This picture is fairly simple as the foreground objects
are well separated from the background, and the objects themselves are seemingly simple shapes at first glance. That said, 
the lighting causes some complex shadows from the snowmen, and the fruit they are supported by. 

The image below shows the original side-by-side with the segmented images produced by the Large (L) and Tiny (T) models.
Randomized colour patches have been used to represent the individual segmentation masks - the individual colours have no 
intrinsic meaning, and it is _not_ important that the same object has different colour masks (e.g. the purple (L) and cyan 
(T) hats).  

[![Snowmen](/data/img/readme/sam-compare-1.png)](/data/img/readme/sam-compare-1.png)

Let's start with the positives: both the Large (L) and Tiny (T) models have done very well at capturing the fruits, supporting
sticks and background sheets of paper. Several properties of the snowmen have been captured well, such as the scarves
(L found both, T only found one), the hats (L: both, T: one), and noses (both L and T got both!). 

However, there were a few sticking points, such as the buttons, since L got 3 out of 4, and T only got one. The eyes and 
arms were completely missed by L - are these perhaps complex shapes? L only managed to capture the left-hand mouth. 

What's also interesting is that, at this scale, the semantic concept of the whole snowman has not been captured, i.e. there is 
no snowman mask like there is for the lemon. You'd have thought it might at least be able to bring out the face. (By the
way, L has a mask for the entire lemon, but also additional masks for the ends, which have been overlaid on top of the main 
mask.)

Finally, I found it interesting that the orange's shadow was not omitted perfectly by either model, whereas the lemon's 
one was by both L and T.

So, whilst the Large model has done reasonably(?) well on the relatively simple snowmen image, let's put it to work on 
another festive image - this time a close-up of a Christmas tree. This image is far more complex, with mixed lighting 
effects, a large variety of shapes and colours, and no obvious background structure. Let's see how SAM2 does. 

[![Trees](/data/img/readme/sam-compare-2.png)](/data/img/readme/sam-compare-2.png)

Well, the Large model has done a pretty good job of picking out most of the hanging decorations - the Santa Claus in the 
top-right (L only), the sleigh with gifts to the left (L & T), the candy cane (L only) and the robin nestled into the tree 
next to it (L only). It has managed to find all four baubles (L & T), and separately classified the visible string for the 
one at the top (L only). The angel was picked out perfectly in her entirety (then her head was given a separate mask) (L & T). 

Even one of our snowmen from the previous image sitting near the bottom of the tree was segmented by the Large model, 
this time in its entirety! Perhaps the size of the object relative the whole image, in addition to its complexity, is 
important for the model as to whether it receives a whole mask. Note that the hat - much like the angel's head - was 
given an additional mask, which was overlaid on top. T completely missed the snowman.

However, there were some complications and notable misses. For example, both models only managed to pick out some parts 
of the sweet wrapper on the right, and the toy Santa in the middle has mostly been missed (though L did get the hat).

Then there are the tree branches - both models have picked out a handful of individual branches, but by no means all of
them. What constitutes a noteworthy branch to the model is unclear and mysterious. We do have the ability to manually 
filter out unwanted artefacts like this for downstream applications, but the inconsistency is strange. 

Finally, the lights play an interesting role in this segmentation - for example, the green light near the centre has been 
picked out by L (it has a reddish mask), but most other lights haven't. There is also a red patch in the top-left corner 
of the image that has been given a mask by L. 

Overall, the Large model has done a good job but with significant caveats.

### Other Features

There are other features of SAM2 that we haven't discussed in this post. For example, you can feed in coordinate-point
prompts (positive and negative) to guide the segmentation process - useful if you have prior knowledge of the objects
in the images. 

SAM2, in contrast to vanilla SAM, can process videos as well - there are some stunning use cases of this feature in Meta's 
original [blog post](https://ai.meta.com/blog/segment-anything-2/). 


### Conclusion

In this post, we studied the SAM2 model in its smallest (Tiny) and largest (Large) incarnations. It is a very powerful 
class of models for segmentation in a relatively compact size - it's possible to run it on a laptop without a GPU. 
We discovered that it is useful for automated segmentation tasks, but the outputs should be examined and checked for 
quality by a human afterwards, especially for complex images (including Christmas trees!).

I also wrote a script for running segmentation with SAM2 on a set of images in a loop - the script is
[here](https://github.com/lukerm/adventures-with-sam/blob/main/sam_src/run_sam2.py).

This mini-project has been a fun learning experience for me, and I hope that you find it interesting and useful too. If 
you have any questions, please don't hesitate to get in touch via the [contact page](https://zl-labs.tech/contact) on the
ZL Labs website.

If you enjoyed reading it, please hit the ⭐ button!

I wish you a very happy holiday season wherever you are in the world and however you celebrate it! 🎄🎅🎁🕎
  
