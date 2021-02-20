#!/bin/python3
import numpy as np
import cv2

scale = 75
verbose = True

def linear_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (gray / 255)**2.2

def hessian(gray):
    dy, dx = np.gradient(gray)
    dyy, dxy = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    return dxx, dxy, dyy

def find_corners(img):
    gray = linear_gray(img)
    gray = cv2.GaussianBlur(gray, (scale, scale), scale/3)
    gray = gray - cv2.GaussianBlur(gray, (3*scale, 3*scale), scale)
    cv2.imshow('gray', gray)
    dxx, dxy, dyy = hessian(gray)
    gray = dxy**2 - 4*dxx*dyy
    gray = cv2.GaussianBlur(gray, (scale, scale), scale/7)
    threshold = np.quantile(gray, 0.99)
    cv2.imshow('discriminant', gray/threshold)
    mask = np.uint8(gray > threshold)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    return np.float32(centroids[1:])

def grid_match(points, directions=4):
    """returns index to neighbors for each point in grid"""
    flann = cv2.flann_Index()
    flann.build(points, {'algorithm': 1, 'trees': 1})
    best_score, best_result = -float("inf"), None
    for m in range(len(points)):
        origin = points[m:m+1,:]
        # neighbors[m, i] = n means that neighbor of points[m] in directions[i] is points[n]
        # last row in the result array represents a "no match" point
        result = -np.ones((points.shape[0] + 1, directions), np.int32)
        score = 0
        for i, n in enumerate(flann.knnSearch(origin, directions+1)[0].T[1:]):
            target = points[n]
            cv2.arrowedLine(img, (round(origin[0,0]), round(origin[0,1])), (round(target[0,0]), round(target[0,1])), (50, 50, 200))
            shifted = points + (target - origin)
            indices, distances = flann.knnSearch(shifted, 1)
            indices[distances > scale] = -1
            result[indices[:,0], i] = np.arange(len(indices))
            score += sum(indices >= 0)
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
    """returns integer grid coordinates for each point in grid"""
    best_score, best_result = 0, None
    visited = np.zeros((neighbors.shape[0],), np.bool)
    magic = -2**31
    while not visited.all():
        result = magic * np.ones((neighbors.shape[0], directions.shape[1]), np.int32)
        corner = np.array((0, 0))
        seed = next(i for i, v in enumerate(visited) if not v)
        flood = [seed]
        result[seed] = 0
        score = 0
        while flood:
            index = flood.pop()
            visited[index] = True
            score += 1
            for n, step in zip(neighbors[index,:], directions):
                if n < 0:
                    continue
                elif result[n,0] == magic:
                    result[n] = result[index] + step
                    corner = np.minimum(corner, result[n])
                    flood.append(n)
                else:
                    if not (result[n] == result[index] + step).all():
                        score = -float("inf")
                    # print(index, result[index], "->", n, result[index] + step, "!=", result[n,:])
        result -= corner
        if score > best_score:
            best_score, best_result = score, result
    return best_result

def calibrate(img):
    imgpoints = find_corners(img)

    if verbose:
        for x, y in imgpoints:
            cv2.circle(img, (round(x), round(y)), 5, (50, 50, 200))
        cv2.imshow('grid', img)
        cv2.waitKey(0)

    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    neighbors = grid_match(imgpoints, len(directions))
    order_cross(neighbors)
    objpoints = grid_positions(neighbors, directions)
    mask = (objpoints[:,0] >= 0)
    imgpoints = imgpoints[mask,:]
    objpoints = objpoints[mask,:]

    if verbose:
        for (x, y), (i, j) in zip(imgpoints, objpoints):
            cv2.circle(img, (round(x), round(y)), 5, (50, 50, 200))
            cv2.putText(img, f"{i}, {j}", (round(x), round(y)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 150, 250))
        cv2.imshow('grid', img)
        cv2.waitKey(0)

    objpoints = np.hstack((objpoints, np.zeros((len(objpoints),1), np.float32))).astype(np.float32)
    shape = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], shape, None, None)
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, 1, shape)
    return dist, mtx, newmtx

if __name__ == "__main__":
    from sys import argv
    from pathlib import Path
    fname = argv[1]
    img = cv2.imread(fname)
    is_portrait = img.shape[0] > img.shape[1]
    # guess grid size, in pixels
    scale = (img.shape[0] + img.shape[1]) / 200
    scale = 2 * round(scale) + 1
    dist, mtx, newmtx = calibrate(img)
    print(dist)
    print(mtx)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newmtx, (img.shape[1], img.shape[0]), 5)
    directory = Path("undistorted")
    directory.mkdir(exist_ok=True)
    for fname in argv[1:]:
        img = cv2.imread(fname)
        path = Path(fname)
        if (img.shape[0] > img.shape[1]) != is_portrait:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite(str(directory / path.name), img)
