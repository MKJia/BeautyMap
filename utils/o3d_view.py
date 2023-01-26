'''
# Created: 2023-1-26 16:38
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

This file is for open3d view control set from view_file, which should be json
1. use normal way to open any gemotry and set view by mouse you want
2. `CTRL+C` it will copy the view detail at this moment.
3. `CTRL+V` to json file
4. give the json file path

Test if you want by run this script

Then press 'V' on keyboard, will set from json

'''

import open3d as o3d
import os, sys
import json

class ViewControl:
    def __init__(self, vctrl: o3d.visualization.ViewControl, view_file=None):
        self.vctrl = vctrl
        self.params = None
        if view_file is not None:
            self.parase_file(view_file)
            self.set_param()
    def read_viewTfile(self, view_file):
        self.parase_file(view_file)
        self.set_param()
    def save_viewTfile(self, view_file):
        return
    def parase_file(self, view_file):
        if(os.path.exists(view_file)):
            with open((view_file)) as user_file:
                file_contents = user_file.read()
                self.params = json.loads(file_contents)
        else:
            print(f"\033[91mDidn't find the file, please check it again: {view_file} \033[0m")
            print(f"NOTE: If you still have this error, please give the absulote path for view_file")
            sys.exit()

    def set_param(self):
        self.vctrl.change_field_of_view(self.params['trajectory'][0]['field_of_view'])
        self.vctrl.set_front(self.params['trajectory'][0]['front'])
        self.vctrl.set_lookat(self.params['trajectory'][0]['lookat'])
        self.vctrl.set_up(self.params['trajectory'][0]['up'])
        self.vctrl.set_zoom(self.params['trajectory'][0]['zoom'])


if __name__ == "__main__":
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # 1. define
    viz = o3d.visualization.VisualizerWithKeyCallback()
    # 2. create
    viz.create_window()
    # 3. add geometry
    viz.add_geometry(pcd)
    # 4. get control !!! must step by step
    ctr = viz.get_view_control()

    o3d_vctrl = ViewControl(ctr)

    def set_view(viz):
        #Your update routine
        o3d_vctrl.read_viewTfile('/home/kin/workspace/EDOMap/data/o3d_view/default_test.json')
        viz.update_renderer()
        viz.poll_events()
        viz.run()

    viz.register_key_callback(ord('V'), set_view)
    viz.run()
    viz.destroy_window()
    print("\033[92mAll o3d_view codes run successfully, Close now..\033[0m See you!")