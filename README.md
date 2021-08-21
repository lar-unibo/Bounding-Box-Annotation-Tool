# Bounding Box Annotation Tool

![gui](figures/gui.png?raw=true)

## 
Tool for drawing bounding boxes on images. 
Input images should be contained inside a ```images``` folder whereas labels are saved in a ```labels``` folder. The file ```classes.txt``` should contain all the classes names (one for each row).

These namening conventions can be configured in the ```params.yaml``` file.


### label format (yolo style):
```
[class_id, center_x, center_y, width, height]
```

## Requirements (major):

- python3
- matplotlib
- PIL
- tkinter

