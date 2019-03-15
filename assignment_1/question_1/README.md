# Data Management - Line Dataset Generation

This problem is aimed at introducing you to the housekeeping around any Machine Learning
problem, Data Management. Data that is not properly labelled becomes hard to manage. You
will learn about numpy array manipulation and basic image handling by solving this problem.

## Problem Statement
Make a dataset of (28 × 28 × 3) images of straight lines on a black background with the
following variations:

* Length - 7 (short) and 15 (long) pixels.
* Width - 1 (thin) and 3 (thick) pixels.
* Angle with X-axis - Angle θ ∈ [0
, 180
) at intervals of 15
.
* Color - Red and Blue.

## Requirements

* numpy: (1.16.2)
* pillow: (5.1.0)
* scipy: (1.2.1)


## Running and understanding of code

The function given below is to rotate the line by intervals of 15 degrees.

```python
def linerot (img,l,w,c,index,index2,index3):
    for i,angle in enumerate(range(0,180,15)):
        rotated=img.rotate(angle, expand=True)
        translation(rotated,i,w,c,l,index,index2,index3)
```

The function gives below is to translate the given lines in 28X28.

```python
def translation(img,i,w,c,l,index,index2,index3):
    for j in range(1,1001):
        background = Image.new('RGB', (28, 28), (0,0,0))
        if(l==7):
            offset=(random.randint(0,16),random.randint(0,16))
            background.paste(img, offset)
            
            background.save('./class/'+str(index2)+'_'+str(index3)+'_'+str(i)+'_'+str(index)+'_'+str(j)+'.jpg')
        else:
            offset=(random.randint(0,8),random.randint(0,8))
            background.paste(img, offset)
            background.save('./class/'+str(index2)+'_'+str(index3)+'_'+str(i)+'_'+str(index)+'_'+str(j)+'.jpg')
        
```


