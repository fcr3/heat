import streamlit as st
import pandas as pd
import numpy as np
import exponential_model as exp_model
import matplotlib.pyplot as plt
import segmentation_util
import streamlit.components.v1 as components
import plotly.express as px

import streamlit_util as st_util
import seaborn as sns

st.title('HEAT')


data_load_state = st.text('Loading data...')

depth_image = np.load('./depth_data/test_depth.npy')
color_image = np.load('depth_data/test_color_image.npy')

xs = []
data = []

for i in range(310, depth_image.shape[0]):
    j = np.argmax(depth_image[i,:])
    if depth_image[i][j] > 0.01:
        xs.append(i)
        data.append(depth_image[i][j])
        
a, b, c ,p, q = exp_model.fit(
    np.array(xs, dtype=np.float64), 
    np.array(data, dtype=np.float64)
)
preds = exp_model.construct_f(a, b, c, p, q)(
    np.array(xs, dtype=np.float64)
)

rows = np.meshgrid(
    np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
)[1]
pred_func = exp_model.construct_f(a, b, c, p, q)
rows = np.meshgrid(
    np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
)[1]
preds_mesh = pred_func(rows)
diff = np.abs(depth_image - preds_mesh)
bool_mat = diff > 200
edit_depth_image = np.copy(depth_image)
edit_depth_image[bool_mat] = 0

edit_color_image = np.copy(color_image)
edit_color_image[diff > 200] = 0


# People segmentation
ppl_seg = segmentation_util.person_segmenter()
ppl_depth = ppl_seg.infer(depth_image, color_image)


# Grab Ground Pointcloud
xyz_gnd = st_util.get_xyz(edit_depth_image)
df_gnd = pd.DataFrame(xyz_gnd, columns=['x', 'y', 'z'])
df_gnd_samp = df_gnd.sample(300)


# Grab People Pointcloud
xyz_ppl = np.array(st_util.get_xyz(ppl_depth))
# xyz_ppl = xyz_ppl[np.argsort(xyz_ppl[:, 2]), :][:10000, :]
df_ppl = pd.DataFrame(xyz_ppl, columns=['x', 'y', 'z'])
df_ppl_samp = df_ppl.sample(300)

df_ppl_sort = df_ppl.sort_values('x', ascending=False).head(300)
df_ppl_sort_mean = df_ppl_sort.mean().values
jittered_pts = np.random.multivariate_normal(
    df_ppl_sort_mean, [[200, 0, 0], [0, 200, 0], [0, 0, 200]], 300
)
df_ppl_sort = pd.DataFrame(jittered_pts, columns=['x', 'y', 'z'])


# Person and Floor Combo Pointcloud
floor_centered = df_gnd_samp - df_gnd_samp.mean()
ppl_flr_center = df_ppl_sort - df_gnd_samp.mean()
ppl_flr_center_show = df_ppl_samp - df_gnd_samp.mean()
ppl_flr_center['type'] = ['ppl'] * 300
ppl_flr_center_show['type'] = ['ppl'] * 300
floor_centered['type'] = ['gnd'] * 300


# Projection Module
_, _, vt = np.linalg.svd((df_gnd_samp - df_gnd_samp.mean()).values)
norm = vt[2]
xyz_ppl_center = (df_ppl_sort - df_gnd_samp.mean()).values
gnd_dists = xyz_ppl_center @ norm
ppl_projs = np.repeat(norm[np.newaxis,...], 300, axis=0) * gnd_dists[...,np.newaxis]
ppl_projs = xyz_ppl_center - ppl_projs

xyz_gnd_center = (df_gnd_samp - df_gnd_samp.mean()).values
gnd_dists2 = xyz_gnd_center @ norm
gnd_projs = np.repeat(norm[np.newaxis,...], 300, axis=0) * gnd_dists2[...,np.newaxis]
gnd_projs = xyz_gnd_center - gnd_projs

ppl_projs_df = pd.DataFrame(ppl_projs, columns=['x', 'y', 'z'])
ppl_projs_df['type'] = ['proj'] * 300
gnd_projs_df = pd.DataFrame(gnd_projs, columns=['x', 'y', 'z'])
gnd_projs_df['type'] = ['gnd'] * 300

# Rotation Module
good_pts = pd.concat([ppl_projs_df, gnd_projs_df])
ppl_pts = good_pts.query(" type == 'proj'")
trans_ppl_pts = ppl_pts.drop(columns=['type']) + df_gnd_samp.mean()
gnd_pts = good_pts.query(" type == 'gnd'")
trans_gnd_pts = gnd_pts.drop(columns=['type']) + df_gnd_samp.mean()

all_plot2 = pd.concat([trans_ppl_pts, trans_gnd_pts]).loc[:, ['x', 'y', 'z']]
_, _, vt2 = np.linalg.svd((all_plot2 - all_plot2.mean()).values)
unitcross = lambda a, b : np.cross(a, b) / np.linalg.norm(np.cross(a, b))
no_xy = np.array(vt2[2], copy=True)
no_xy[0] = 0
no_xy[1] = 0
x, y, z = unitcross(vt2[2], no_xy)

nvt, nx = np.linalg.norm(vt2[2]), np.linalg.norm(no_xy)
cos_ang = np.dot(vt2[2], no_xy) / (nvt * nx)
c = cos_ang
s = np.sqrt(1-c*c)
C = 1-c
rmat = np.array([[x*x*C+c, x*y*C-z*s, x*z*C+y*s ],
                 [y*x*C+z*s, y*y*C+c, y*z*C-x*s ],
                 [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])

xyz = (all_plot2 - all_plot2.mean()).values

rot_xyz = rmat @ (xyz.T)
rot_xyz = rot_xyz.T
rot_xyz[:, 2] = 0
rot_xyz_df = pd.DataFrame(rot_xyz, columns=['x', 'y', 'z'])
rot_xyz_df['type'] = good_pts['type'].values


# Big Module Finale
rot_xyz2, num = st_util.xy_heatmap(ppl_depth, edit_depth_image)
rot_xyz2_df = pd.DataFrame(rot_xyz2, columns=['x', 'y', 'z'])
rot_xyz2_df['type'] = np.concatenate(
    [np.array(['gnd'] * num), np.array(['ppl'] * (rot_xyz2.shape[0] - num))]
)

data_load_state.text('Loading data...done!')

# Render on website

st.write("HEAT uses depth camera data to calculate occupancy statistics in smart spaces!")

st.subheader('Original Data from Depth Camera')
st.write("A camera like a D435 is able to give us both depth data and an associated RGB image. See examples below")

plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 1)
plt.imshow(color_image)
plt.title("Original Color Image")
plt.subplot(1, 2, 2)
plt.imshow(depth_image)
plt.title("Depth Data")
st.pyplot()

st.subheader('Sum of Exponential Model = Ground Plane Detection')
st.write("""
The Ground Plane Detection Model:
""")
st.latex("y = a + be^{px} + ce^{qx}")

st.markdown("""
This model comes from this [paper](https://www.researchgate.net/publication/280882993_Ground_Plane_Detection_Using_an_RGB-D_Sensor).

We fit this model by doing [this](https://fr.scribd.com/doc/14674814/Regressions-et-equations-integrales). If you can read French, then this is the paper for you! If not, just refer to this [post](https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials).

The original paper for ground plane detection tells us to do the following.
""")

components.html("""
Ground Plane Detection Algorithm:
<ol>
    <li>Take the max depth per row of the depth matrix</li>
    <li>Fit the double exponential model to this data. x = the row index.</li>
    <li>Compare model to depth image columns. Large deviations = not ground</li>
</ol>
""")

plt.figure(figsize=(10, 3))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 1)
depth_max_plot = plt.plot(xs, data)
plt.title("Max Depth per Row")
plt.subplot(1, 2, 2)
depth_max_preds = plt.plot(xs, preds)
depth_max_plot2 = plt.plot(xs, data)
plt.title("Sum of Exp Fit Result")

st.pyplot()

st.write('Here are the segmented ground pixels from analyzing deviations over a certain threshold: 200')

plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 1)
plt.imshow(color_image)
plt.title("Original Color Image")
plt.subplot(1, 2, 2)
plt.imshow(edit_depth_image, alpha=1)
plt.title("Ground Depths")
                           
st.pyplot()

st.write("Here is a pointcloud sample that contains 300 points.")
fig = px.scatter_3d(df_gnd_samp, x='x', y='y', z='z')
st.plotly_chart(fig)

st.subheader('Segmenting Person Pixels via OpenVINO')
st.write('Now, we use OpenVINO\'s given segmentation network to get people pixels from the color image.')

plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 1)
plt.imshow(color_image)
plt.title("Original Color Image")
plt.subplot(1, 2, 2)
plt.imshow(ppl_depth, alpha=1)
plt.title("People Depths")
                           
st.pyplot()

st.write("Here is a pointcloud sample that contains 300 points.")
fig = px.scatter_3d(df_ppl_samp, x='x', y='y', z='z')
st.plotly_chart(fig)


st.subheader('Combining Pointclouds for Visualization')
st.write('As a side note, this point cloud is centered around the mean of the ground point cloud. More information as to why in the next section.')

fig = px.scatter_3d(pd.concat([ppl_flr_center_show, floor_centered]), 
                    x='x', y='y', z='z', color='type')
st.plotly_chart(fig)


st.subheader('Project to Ground and Cloudpoint Rotation for Visualization')
st.write("""
In order to get the heatmap, we will project the person pixels on the ground floor pixels. But in order to actually do this projection, we need to use the normal vector of the ground plane. We can do so by getting the third singular vector from the SVD of the ground plane pointcloud. We also clean up the ground pixels such that they reside on one plane. 

Here we begin to lose the actual depth data for the sake of easy visualiztion, but our approximations are based on a theoretically ground plane. In reality, the ground is not that flat!

Also note that the person pointcloud seems to have dramatically condensed. As a processing step, we take the bottom most person depth data in regard to the frame view. Some configuration will be needed if there is roll to the camera. Then we get the mean of those bottom most points to get a representative point of a person. For visualization, we draw from a multivariate guassian distribution to make a mini pointcloud centered around the mean.
""")

fig = px.scatter_3d(pd.concat([ppl_projs_df, gnd_projs_df]), x='x', y='y', z='z', color='type')
st.plotly_chart(fig)

st.write("Next, we rotate the projected point cloud mesh onto the xy plane. Our mesh now does not have any physical meaning but it does paint a picture of where people are relative to each of the objects. The true depths can be sent to other applications.")

fig = px.scatter_3d(rot_xyz_df, x='x', y='y', z='z', color='type')
st.plotly_chart(fig)


st.subheader('Wrapping Up')
st.write("""
As a final point, we will now wrap up the steps into one module. This module takes a depth image and color image assoicated with the depth and outputs (1) the xyz map and (2) the index separating the ground pixels from the row pixels.
""")

st.write("Input:")
plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 2, 1)
plt.imshow(color_image)
plt.title("Original Color Image")
plt.subplot(1, 2, 2)
plt.imshow(depth_image)
plt.title("Depth Data")
st.pyplot()

st.write("Output:")
fig = px.scatter_3d(rot_xyz2_df, x='x', y='y', z='z', color='type')
st.plotly_chart(fig)
# ax = sns.scatterplot(x="x", y="y", hue="type", data=rot_xyz2_df)
# st.pyplot()
         



