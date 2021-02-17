# Rectify Checkerboard

Undistort / rectify lab photos so that distances can be measured precisely

## How to proceed

1) Print a black and white checkerboard with precise dimensions (such as [checkerboard.pdf](checkerboard.pdf)).
2) Put the printed checkerboard on solid flat surface.
3) Build the camera rig with a tripod and take a photo of the checkerboard.
4) Take photos of the objects you want to measure.
5) Run the tool:
`python undistort.py first-photo.jpg measured-object-1.jpg [measured-object-2.jpg ...]`
All the photos will be automatically rectified and stored in a new directory named `undistorted`.
Original photos are preserved.

## Camera recommendations

* Use telephoto / zoom lens and take shots from a distance to reduce parallax.
* Always use a remote control shutter if you can, to reduce motion blur.
* If you have many objects to measure, use a cell phone app for visible numbering on the photos. The built-in timer app is perfect for this purpose.
