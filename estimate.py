#!/bin/python3
import numpy as np
import cv2
from sys import argv

scale = 15

def find_corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (scale, scale), scale/5)
    gray = np.abs(np.diff(np.diff(gray), axis=0)) / 10
    threshold = np.quantile(gray, 0.999)
    mask = (gray > threshold).astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # ignore 0-th component because it is the background
    return centroids[1:].astype(np.float32)

def grid_match(points, directions=4):
    flann = cv2.flann_Index()
    flann.build(points, {'algorithm': 1, 'trees': 1})
    best_score = 0
    best_result = None
    for guess in range(len(points)):
        origin = points[guess:guess+1,:]
        # last row in the result array represents a "no match" point
        result = -np.ones((points.shape[0] + 1, directions), np.int32)
        score = 0
        for i, n in enumerate(flann.knnSearch(origin, directions+1)[0].T[1:]):
            if n == guess:
                score = -float("inf")
            target = points[n]
            shifted = points + (target - origin)
            indices, distances = flann.knnSearch(shifted, 1)
            indices[distances > scale**2] = -1
            result[indices[:,0], i] = np.arange(len(indices))
            score += sum(distances < scale**2)
        score -= sum(sum(result[:,i] == result[:,j]) for i in range(1,4) for j in range(i))
        if score > best_score:
            best_score, best_result = score, result
    return best_result[:-1]

def order_cross(indices):
    """indices: array(n, 4), each column are neighbor indices in a direction
    Ensures that the directions are ordered something like left, right, up, down
    <=> indices[:,0] is inverse permutation of indices[:,1]
    """
    n = indices
    current = sum(n[n[:,0],1] == np.arange(len(n))) + sum(n[n[:,2],3] == np.arange(len(n)))
    swapped = sum(n[n[:,0],2] == np.arange(len(n))) + sum(n[n[:,1],3] == np.arange(len(n)))
    if swapped > current:
        indices[:,1:3] = indices[:,2:0:-1]

def grid_positions(neighbors, directions):
    result = np.zeros((neighbors.shape[0], directions.shape[1]), np.int32)
    visited = np.zeros((neighbors.shape[0],), np.bool)
    while not visited.all():
        seed = next(i for i, v in enumerate(visited) if not v)
        boundary = [seed]
        while boundary:
            index = boundary.pop()
            visited[index] = True
            for n, step in zip(neighbors[index,:], directions):
                if n < 0:
                    continue
                elif not visited[n]:
                    result[n,:] = result[index] + step
                    boundary.append(n)
                else:
                    if not (result[n,:] == result[index] + step).all():
                        print(index, result[index], "->", n, result[index] + step, "!=", result[n,:])
    return result

def calibrate(img):
    imgpoints = find_corners(img)
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    neighbors = grid_match(imgpoints, directions.shape[0])
    order_cross(neighbors)
    objpoints = grid_positions(neighbors, directions)
    objpoints = np.hstack((objpoints, np.zeros((len(objpoints),1), np.float32))).astype(np.float32)
    shape = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], shape, None, None)
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, 1, shape)
    return dist, mtx, newmtx

fname = argv[1]
img = cv2.imread(fname)
dist, mtx, newmtx = calibrate(img)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newmtx, (img.shape[1], img.shape[0]), 5)
for fname in argv[2:]:
    img = cv2.imread(fname)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    basename = fname.rsplit("/", 1)[1]
    cv2.imwrite(f"undistorted/{basename}", img)

# print(mtx)
# print(dist)
# dist = 2 * dist[0,:] / min(shape)
# print(f'-fx "ii = i - $cx; jj = j - $cy; xx = $cos*ii +$sin*jj + $cx; yy = -$sin*ii +$cos*jj + $cy; v.p{{xx,yy}}"')
# print(f'-distort Barrel "{dist[4]} {dist[1]} {dist[0]} 1.0"')

# for (x, y), (i, j, _) in zip(imgpoints, objpoints):
    # cv2.putText(img, f"{i}, {j}", (round(x), round(y)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 150, 250))

# for x, y in centroids:
    # cv2.circle(img, (round(x), round(y)), scale//3, (50, 50, 200))

# for (x1, y1), (index,), (distance,) in zip(shifted, indices, distances):
    # if distance < scale**2:
        # x2, y2 = points[index]
        # cv2.line(img, (round(x1), round(y1)), (round(x2), round(y2)), (50, 50, 200))

# img[:-1,:-1,1] += (500*gray).astype(np.uint8)

# cv2.imshow('img', img)
# cv2.waitKey(0)
