import asyncio
from asyncio import base_events
asyncio.Server = getattr(base_events, 'Server', None)
import sunpy.map

'''
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
PACKAGES
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import astropy.units as u
from astropy.visualization import ImageNormalize, SqrtStretch
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.ndimage import map_coordinates
from scipy.io import readsav
from scipy.signal import find_peaks
from astropy.io import fits

import sunpy.map
import sunpy.visualization.colormaps as cm


'''
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
FUNCTIONS
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
'''

def initCube(year, base_dir):
    path = base_dir / f'AIA{year}_171_Cutout'
    files = sorted(Path(path).glob("*.fits"))
    empty_cube = []
    for i in range(len(files)):
        print(f"Processing frame {i} out of", len(files), end="\r")
        data = sunpy.map.Map(files[i]).data
        empty_cube.append(data)
    cube = np.asarray(empty_cube, dtype=float)
    #cubeFile = open(f'{base_dir}/AIA{year}_171_Cutout/cube_{year}.npy', 'wb')
    #np.save(cubeFile, cube)
    #print(f"Cube saved with dimensions {cube.shape}")
    return cube

def timedistance_norm(r):
    """Return a unit vector perpendicular to the slit direction."""
    rx, ry = r

    if rx == 0:
        return np.array([1.0, 0.0])
    if ry == 0:
        return np.array([0.0, 1.0])

    nx = np.sqrt(ry**2 / (rx**2 + ry**2))
    ny = -rx * nx / ry
    return np.array([nx, ny])


def timedistance(img, coord, width=1):
    """
    Compute a time-distance map along a slit.

    Parameters
    ----------
    img : ndarray
        3D data cube with shape (nt, ny, nx)
    coord : array-like
        [x1, y1, x2, y2] slit coordinates in pixels
    width : int
        Odd slit width in pixels

    Returns
    -------
    td_map : ndarray
        2D array with shape (time, distance)
    """
    x1, y1, x2, y2 = coord
    nt, ny, nx = img.shape

    n = int(round(np.hypot(x2 - x1, y2 - y1))) + 1
    x = np.linspace(x1, x2, n)
    y = np.linspace(y1, y2, n)

    # Width = 1: just sample along the slit
    if width == 1:
        td_map = np.zeros((nt, n))
        for frame in range(nt):
            td_map[frame, :] = map_coordinates(
                img[frame, :, :],
                [y, x],
                order=1,
                mode='nearest'
            )
        return td_map

    # Width > 1: average over neighbouring slits
    r = np.array([x2 - x1, y2 - y1])
    n_hat = timedistance_norm(r)
    k_vals = np.arange(-(width // 2), width // 2 + 1)

    td_map = np.zeros((nt, n))

    for frame in tqdm(range(nt)):
        frame_vals = np.zeros((width, n))

        for i, k in enumerate(k_vals):
            x_off = x + k * n_hat[0]
            y_off = y + k * n_hat[1]

            frame_vals[i, :] = map_coordinates(
                img[frame, :, :],
                [y_off, x_off],
                order=1,
                mode='nearest'
            )

        td_map[frame, :] = np.mean(frame_vals, axis=0)

    return td_map

# Main year selector for this notebook
save_path='/Users/smolp/OneDrive/Documents/Uni Work/PhD/Sunpy Testing/AIA171_Cutout'
year = '2023'

# Update this base directory only if your data location changes
base_dir = Path('/Users/smolp/OneDrive/Documents/Uni Work/PhD/Sunpy Testing/')
path = base_dir / f'AIA{year}_171_Cutout'

files = sorted(Path(path).glob("*.fits"))

with fits.open(files[0]) as hdul:
    hdul.info()
    # print(hdul[0].header)
    # print(hdul[1].header)
    for meta in hdul[1].header:
        print(meta, hdul[1].header[meta])

# empty_cube = []
# for i in range (0,len(files)):
#     data = sunpy.map.Map(files[i]).data
#     empty_cube.append(data)

if (Path(f'{base_dir}/AIA{year}_171_Cutout/cube_{year}.npy').exists()):
    cube = np.load(f'{base_dir}/AIA{year}_171_Cutout/cube_{year}.npy')
else:
    print(f'Cube file not found. Initializing cube for year {year}...')
    cube = initCube(year, base_dir)

cube = np.asarray(cube, dtype=float)

for path in files:
    hdul = fits.open(path)

m = sunpy.map.Map(files[0])
data = m.data
print(data)

#file = readsav(path)
#cube = np.asarray(file.data, dtype=float)

#print(f'Loaded: {path}')
#print(f'Cube shape: {cube.shape}')


#first_frame = cube[0]
first_frame = data

fig = plt.figure(figsize=(7, 7))

norm = ImageNormalize(
    vmin=np.nanpercentile(first_frame, 1),
    vmax=np.nanpercentile(first_frame, 99),
    stretch=SqrtStretch()
)

plt.imshow(first_frame, origin='lower', norm=norm, cmap=cm.cmlist['sdoaia171'])
plt.title(f'{year}: cropped first frame')
plt.xlabel('Solar X [pixels]')
plt.ylabel('Solar Y [pixels]')
plt.show()



# ----------------------------
# CROP REGION (centre + width)
# ----------------------------
xc = 418   # x centre [pixels]
yc = 418   # y centre [pixels]

dx = 835    # full width in x [pixels]
dy = dx    # full width in y [pixels]

x0 = int(xc - dx // 2)
x1 = int(xc + dx // 2)
y0 = int(yc - dy // 2)
y1 = int(yc + dy // 2)

# Keep crop inside the image boundaries
x0 = max(0, x0)
y0 = max(0, y0)
x1 = min(cube.shape[2], x1)
y1 = min(cube.shape[1], y1)

crop_cube = cube[:, y0:y1, x0:x1]

print(f'Crop bounds: x={x0}:{x1}, y={y0}:{y1}')
print(f'Cropped cube shape: {crop_cube.shape}')

first_frame = crop_cube[0]

fig = plt.figure(figsize=(7, 7))

norm = ImageNormalize(
    vmin=np.nanpercentile(first_frame, 1),
    vmax=np.nanpercentile(first_frame, 99),
    stretch=SqrtStretch()
)

plt.imshow(first_frame, origin='lower', norm=norm, cmap=cm.cmlist['sdoaia171'])
plt.title(f'{year}: cropped first frame')
plt.xlabel('Solar X [pixels]')
plt.ylabel('Solar Y [pixels]')
plt.show()


# --------------------------------------------------
# SETUP IMAGE
# --------------------------------------------------
fig, (ax_img, ax_td) = plt.subplots(1, 2, figsize=(10, 5))
#%matplotlib widget
norm = ImageNormalize(
    vmin=np.nanpercentile(first_frame, 1),
    vmax=np.nanpercentile(first_frame, 99),
    stretch=SqrtStretch()
)

ax_img.imshow(first_frame, origin='lower', norm=norm,
              cmap=cm.cmlist['sdoaia171'])

ax_img.set_title("Click 2 points to define slit")
ax_td.set_title("Time-Distance Map")

points = []
previous_lines = []
previous_points = []
previous_labels = []
slit_counter = 0
slit_line = None
td_im = None

def compute_td(x1, y1, x2, y2):
    coord = np.array([x1, y1, x2, y2], dtype=float)

    td_map = timedistance(crop_cube, coord, width=3)

    # # optional quick detrend for nicer display
    # window_length = 31
    # if window_length % 2 == 0:
    #     window_length += 1

    # td_smooth = savgol_filter(td_map, window_length, 2, axis=0)
    # td_detrended = td_map - td_smooth

    return td_map

def update_td(td_map):
    global td_im

    ax_td.clear()

    td_im = ax_td.imshow(
        td_map.T,
        origin='lower',
        aspect='auto',
        cmap='gray'
    )

    ax_td.set_title("Time-Distance Map")
    ax_td.set_xlabel("Time [frames]")
    ax_td.set_ylabel("Distance [pixels]")

    fig.canvas.draw_idle()

def onclick(event):
    global slit_line, points, slit_counter
    global previous_lines, previous_points, previous_labels

    if event.inaxes != ax_img:
        return

    if event.xdata is None or event.ydata is None:
        return

    # store point
    points.append((event.xdata, event.ydata))

    # plot current click (red)
    pt, = ax_img.plot(event.xdata, event.ydata, 'ro')
    previous_points.append(pt)

    # wait for 2 clicks
    if len(points) == 2:

        x1, y1 = points[0]
        x2, y2 = points[1]

        # ----------------------------------
        # FADE ALL PREVIOUS SLITS + POINTS
        # ----------------------------------
        for line in previous_lines:
            line.set_color('gray')
            line.set_alpha(0.5)

        for pt in previous_points[:-2]:  # exclude current two points
            pt.set_color('gray')
            pt.set_alpha(0.5)

        for label in previous_labels:
            label.set_color('gray')
            label.set_alpha(0.5)

        # ----------------------------------
        # DRAW NEW SLIT (RED)
        # ----------------------------------
        slit_line, = ax_img.plot([x1, x2], [y1, y2], 'r-', lw=2)
        previous_lines.append(slit_line)

        # ----------------------------------
        # ADD SLIT NUMBER
        # ----------------------------------
        slit_counter += 1
        xm = x2+3
        ym = y2+3

        label = ax_img.text(
            xm, ym,
            str(slit_counter),
            color='red',
            fontsize=10,
            ha='center',
            va='center',
            weight='bold'
        )
        previous_labels.append(label)

        # ----------------------------------
        # COMPUTE TD
        # ----------------------------------
        td_map = compute_td(x1, y1, x2, y2)
        update_td(td_map)

        print(f"Slit {slit_counter} updated")

        # reset for next slit
        points = []

        fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

finalslit = previous_lines[-1].get_data()

'''
Finalslit[0] = x coordinates
Finalslit[1] = y coordinates   
'''

xmin = finalslit[0][0]
xmax = finalslit[0][1]
ymin = finalslit[1][0]
ymax = finalslit[1][1]

# ----------------------------
# CROP REGION (centre + width)
# ----------------------------
xc = ((xmax + xmin) // 2)   # x centre [pixels]
yc = ((ymax + ymin) // 2)   # y centre [pixels]

if xmax - xmin > ymax - ymin:
    dx = (xmax - xmin) * 2
    dy = dx
else:
    dy = (ymax - ymin) * 2
    dx = dy
# dx = 835    # full width in x [pixels]
# dy = dx    # full width in y [pixels]

print(f"xmin: {xmin}")
print(f"ymin: {ymin}")
print(f"xmax: {xmax}")
print(f"ymax: {ymax}")
print(f"dx: {dx}")
print(f"dy: {dy}")
print(f"xc: {xc}")
print(f"yc: {yc}")

x0 = int(xc - dx // 2)
x1 = int(xc + dx // 2)
y0 = int(yc - dy // 2)
y1 = int(yc + dy // 2)

# Keep crop inside the image boundaries
x0 = max(0, x0)
y0 = max(0, y0)
x1 = min(cube.shape[2], x1)
y1 = min(cube.shape[1], y1)

crop_cube = cube[:, y0:y1, x0:x1]

print(f'Crop bounds: x={x0}:{x1}, y={y0}:{y1}')
print(f'Cropped cube shape: {crop_cube.shape}')

first_frame = crop_cube[0]

fig = plt.figure(figsize=(7, 7))

norm = ImageNormalize(
    vmin=np.nanpercentile(first_frame, 1),
    vmax=np.nanpercentile(first_frame, 99),
    stretch=SqrtStretch()
)

plt.imshow(first_frame, origin='lower', norm=norm, cmap=cm.cmlist['sdoaia171'])
plt.title(f'{year}: cropped first frame')
plt.xlabel('Solar X [pixels]')
plt.ylabel('Solar Y [pixels]')
plt.show()

plotfull = input("Do you want to plot the full animation of the cropped region? (y/n): ")
if plotfull.lower() == 'y':
    fig, ax = plt.subplots(figsize=(7, 7))
    norm = ImageNormalize(
        vmin=np.nanpercentile(first_frame, 1),
        vmax=np.nanpercentile(first_frame, 99),
        stretch=SqrtStretch()
    )

    for i in range(crop_cube.shape[0]):
        ax.clear()
        ax.imshow(crop_cube[i], origin='lower', norm=norm, cmap=cm.cmlist['sdoaia171'])
        ax.set_title(f'Frame {i+1}/{crop_cube.shape[0]}')
        ax.set_xlabel('Solar X [pixels]')
        ax.set_ylabel('Solar Y [pixels]')
        plt.pause(0.005)  # pause to create animation effect
    plt.show()

savecube = input("Do you want to save the cropped cube as a .npy file? (y/n): ")
if savecube.lower() == 'y':
    savename = str(input(f"Save as:{base_dir}/AIA{year}_171_Cutout/"))
    np.save(f"{base_dir}/AIA{year}_171_Cutout/{savename}.npy", crop_cube)
    print(f"Cropped cube saved as {savename}.npy")

# points = []
# #%matplotlib widget
# fig, ax = plt.subplots(figsize=(8, 8))

# norm = ImageNormalize(
#     vmin=np.nanpercentile(first_frame, 1),
#     vmax=np.nanpercentile(first_frame, 99),
#     stretch=SqrtStretch()
# )

# ax.imshow(
#     first_frame,
#     origin='lower',
#     norm=norm,
#     cmap=cm.cmlist['sdoaia171']
# )

# ax.set_title("Click 10 points (5 slit pairs)")
# ax.set_xlabel("Solar X [pixels]")
# ax.set_ylabel("Solar Y [pixels]")

# # --- click handler ---
# def onclick(event):
#     if event.xdata is None or event.ydata is None:
#         return

#     points.append((event.xdata, event.ydata))
#     ax.plot(event.xdata, event.ydata, 'ro')
#     fig.canvas.draw()

#     print(f"Point {len(points)}: ({event.xdata:.1f}, {event.ydata:.1f})")

#     if len(points) == 10:
#         fig.canvas.mpl_disconnect(cid)
#         draw_slits()

# def draw_slits():
#     slit_pairs = [(points[i], points[i+1]) for i in range(0, 10, 2)]

#     for i, ((x1, y1), (x2, y2)) in enumerate(slit_pairs):
#         ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
#         ax.plot([x1, x2], [y1, y2], 'ro')

#         dx = x2 - x1
#         dy = y2 - y1
#         length = np.hypot(dx, dy)

#         if length > 0:
#             lx = x2 + 5 * dx / length
#             ly = y2 + 5 * dy / length
#             ax.text(
#                 lx, ly, f"{i+1}",
#                 color='black',
#                 fontsize=12,
#                 weight='bold',
#                 ha='center',
#                 va='center'
#             )

#     save_path="/Users/smolp/OneDrive/Documents/Uni Work/PhD/Sunpy Testing/"
#     plt.savefig(f"{save_path}fig3-{year}_Slit_Positions.png", dpi=200, bbox_inches="tight")
#     plt.show()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()


# #%matplotlib inline

# cadence = 12.0   # seconds
# width = 5        # slit width (pixels)

# td_maps = []
# td_maps_detrended = []

# # Build slit pairs from clicked points
# slit_pairs = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
# if len(slit_pairs) != 5:
#     raise ValueError(f"Expected 5 slits, got {len(slit_pairs)}")

# # Plot setup
# fig_td, axes_td = plt.subplots(5, 2, figsize=(12, 12), sharex=True, constrained_layout=True)

# # Loop over slits
# for i, ((x1, y1), (x2, y2)) in enumerate(slit_pairs):

#     coord = np.array([x1, y1, x2, y2], dtype=float)
#     td_map = timedistance(crop_cube, coord, width=width)
#     td_maps.append(td_map)

#     # Quick detrend
#     window_length = 31 + (31 % 2 == 0)
#     td_smooth = savgol_filter(td_map, window_length=window_length, polyorder=2, axis=0)
#     td_detrended = td_map - td_smooth
#     td_maps_detrended.append(td_detrended)

#     # Plot
#     ax1, ax2 = axes_td[i]

#     ax1.imshow(td_map.T, origin='lower', aspect='auto', cmap=cm.cmlist['sdoaia171'])
#     ax1.set_title(f"TD map (slit {i+1})")
#     ax1.set_ylabel("Distance [pixels]")

#     ax2.imshow(td_detrended.T, origin='lower', aspect='auto', cmap='gray')
#     ax2.set_title(f"Detrended TD map (slit {i+1})")
#     ax2.set_ylabel("Distance [pixels]")

# axes_td[-1, 0].set_xlabel("Time [frames]")
# axes_td[-1, 1].set_xlabel("Time [frames]")

# plt.show()


# map_1 = td_maps[0]
# map_2 = td_maps[1]
# map_3 = td_maps[2]
# map_4 = td_maps[3]
# map_5 = td_maps[4]

# print("TD map shapes:", map_1.shape, map_2.shape, map_3.shape, map_4.shape, map_5.shape)


# def detrend_td_map(td_map, window_length_s, cadence, polyorder=2):
    
#     # Detrend a time-distance map using Savitzky–Golay filtering.

#     # Parameters
#     # ----------
#     # td_map : 2D array
#     #     Time-distance map (time x distance or vice versa depending on your convention)
#     # window_length_s : float
#     #     Smoothing window in seconds
#     # cadence : float
#     #     Time step per frame in seconds
#     # polyorder : int
#     #     Polynomial order for Savitzky–Golay filter

#     # Returns
#     # -------
#     # detrended : 2D array
#     #     td_map with smooth trend removed
#     # smooth : 2D array
#     #     The smooth background trend
    

#     # convert seconds → frames
#     window_length = int(window_length_s / cadence)

#     # must be odd
#     if window_length % 2 == 0:
#         window_length += 1

#     # safety check (must be > polyorder)
#     if window_length <= polyorder:
#         window_length = polyorder + 2 + (polyorder % 2)

#     smooth = savgol_filter(
#         td_map,
#         window_length=window_length,
#         polyorder=polyorder,
#         axis=0   # IMPORTANT: smoothing along time axis
#     )

#     detrended = td_map - smooth

#     return detrended, smooth

# # DETREND MULTIPLE TD MAPS
# detrend_window = 10 * 60   # seconds
# polyorder = 2

# td_list = [map_1, map_2, map_3, map_4, map_5]

# detrended_maps = []
# smooth_maps = []

# for i, td in enumerate(td_list):

#     detrended, smooth = detrend_td_map(
#         td,
#         window_length_s=detrend_window,
#         cadence=cadence,
#         polyorder=polyorder
#     )

#     detrended_maps.append(detrended)
#     smooth_maps.append(smooth)

#     print(f"Slit {i+1} detrended")

# # optional unpack
# detrended_map_1, detrended_map_2, detrended_map_3, detrended_map_4, detrended_map_5 = detrended_maps
# smooth_map_1, smooth_map_2, smooth_map_3, smooth_map_4, smooth_map_5 = smooth_maps


# # ----------------------------
# # PLOT: 3 SLITS × (ORIGINAL, SMOOTH, DETRENDED)
# # ----------------------------
# fig, axes = plt.subplots(5, 3, figsize=(12, 10), constrained_layout=True)

# slits = [1, 2, 3, 4, 5]

# for i in range(5):
#     og = td_list[i]
#     sm = smooth_maps[i]
#     dt = detrended_maps[i]

#     # ---- ORIGINAL ----
#     axes[i, 0].imshow(
#         og.T,
#         origin='lower',
#         aspect='auto',
#         cmap=cm.cmlist['sdoaia171'], 
#         vmin = np.nanpercentile(og, 1),
#         vmax = np.nanpercentile(og, 99)
#     )
#     axes[i, 0].set_ylabel(f'Slit {slits[i]}\nDistance [pixels]')

#     # ---- SMOOTH ----
#     axes[i, 1].imshow(
#         sm.T,
#         origin='lower',
#         aspect='auto',
#         cmap='gray', 
#         vmin = np.nanpercentile(sm, 1),
#         vmax = np.nanpercentile(sm, 99)
#     )

#     # ---- DETRENDED ----
#     axes[i, 2].imshow(
#         dt.T,
#         origin='lower',
#         aspect='auto',
#         cmap='gray', 
#         vmin = np.nanpercentile(dt, 1),
#         vmax = np.nanpercentile(dt, 99)
#     )

#     # X labels only on bottom row (cleaner)
#     if i == 2:
#         for j in range(3):
#             axes[i, j].set_xlabel('Time [frames]')
    
#     axes[0, 0].set_title('Original')
#     axes[0, 1].set_title('Smooth')
#     axes[0, 2].set_title('Detrended')
#     save_path="/Users/smolp/OneDrive/Documents/Uni Work/PhD/Sunpy Testing/"
#     plt.savefig(f"{save_path}fig4-{year}_Map_Detrending.png", dpi=200, bbox_inches="tight")
# plt.show()
