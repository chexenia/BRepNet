
from random import random
import numpy as np
import math 
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

from pipeline.entity_mapper import EntityMapper

from occwl.io import load_step
from occwl.compound import Compound
from occwl.face import Face
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Solid, TopoDS_Face

from OCC.Extend.DataExchange import read_step_file_with_names_colors, list_of_shapes_to_compound

def load_step_face_named(filename, as_compound=True):
    from OCC.Core.STEPControl import (
        STEPControl_Reader,
    )
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
    from OCC.Core.StepRepr import StepRepr_RepresentationItem as hsr

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == IFSelect_RetDone:  # check status
        transfer_result = step_reader.TransferRoots()
        if not transfer_result:
            raise AssertionError("Transfer failed.")
        _nbs = step_reader.NbShapes()
        if _nbs == 0:
            raise AssertionError("No shape to transfer.")
        if _nbs == 1:  # most cases
            shape = step_reader.Shape(1)
            tr = step_reader.WS().TransferReader()
            
            # Explore the faces of the shape (these are known to be named)
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            face_dict = dict()
            while exp.More():
                s = exp.Current()
                exp.Next()
                item = tr.EntityFromShapeResult(s, -1)
                if item:
                    item = hsr.DownCast(item)
                    name = item.Name().ToCString()
                    face_dict[Face(s)] = name

                    if name:
                        print('Found entity named: {}: {}.'.format(name, s))
                else:
                    raise RuntimeError(f"Could not find entity for shape{s}")

            return list(Compound(shape).solids()), face_dict

        if _nbs > 1:
            print("Number of shapes:", _nbs)
            shps = []
            # loop over root shapes
            for k in range(1, _nbs + 1):
                new_shp = step_reader.Shape(k)
                if not new_shp.IsNull():
                    shps.append(new_shp)
            if as_compound:
                compound, result = list_of_shapes_to_compound(shps)
                if not result:
                    print("Warning: all shapes were not added to the compound")
                return compound
            print("Warning, returns a list of shapes.")
            return shps, None
    else:
        raise AssertionError("Error: can't read file.")
    return None, None

def load_step_face_colored(inpath):
    """
    Load a STEP file and return a list of faces with colors
    """
    shape_dict = read_step_file_with_names_colors(str(inpath))

    shp = []
    face_dict = dict()
    for a_shape in shape_dict:
        l, c = shape_dict[a_shape]
        shp.append(a_shape)
        face_dict[a_shape] = c
        if isinstance(a_shape, TopoDS_Solid):
            shp.append(a_shape)
        elif isinstance(a_shape, TopoDS_Face):
            color = [c.Red(), c.Green(), c.Blue()]
            f = Face(a_shape)
            face_dict[f] = color

    if not isinstance(shp, TopoDS_Compound):
            shp, success = list_of_shapes_to_compound(shp)
            assert success
    return list(Compound(shp).solids()), face_dict
    
class ColorMap:
    def __init__(self):
        self.color_values = [
            [255, 0, 0],  # Blue
            [0, 255, 0],  # Green
            [0, 0,255]    # Red
        ]

    def interpolate_value(self, a, b, t):
        return (b-a)*t + a

    def interpolate_color(self, t):
        num_colors = len(self.color_values)
        tp = t*(num_colors-1)
        index_before = math.floor(tp)
        index_after = math.ceil(tp)
        tint = tp-index_before
        color = []
        for i in range(3):
            color.append(
                self.interpolate_value(
                    self.color_values[i][index_before], 
                    self.color_values[i][index_after], 
                    tint
                )
            )
        return color

class MultiSelectJupyterRenderer(JupyterRenderer):
    def __init__(self, *args, **kwargs):
        super(MultiSelectJupyterRenderer, self).__init__(*args, **kwargs)
            
    def click(self, value):
        """ called whenever a shape  or edge is clicked
        """
        try:
            obj = value.owner.object
            self.clicked_obj = obj
            if self._current_mesh_selection != obj:
                if obj is not None:
                    self._shp_properties_button.disabled = False
                    self._toggle_shp_visibility_button.disabled = False
                    self._remove_shp_button.disabled = False
                    id_clicked = obj.name  # the mesh id clicked
                    self._current_mesh_selection = obj
                    self._current_selection_material_color = obj.material.color
                    obj.material.color = self._selection_color
                    # selected part becomes transparent
                    obj.material.transparent = True
                    obj.material.opacity = 0.5
                    # get the shape from this mesh id
                    selected_shape = self._shapes[id_clicked]
                    self._current_shape_selection = selected_shape
                # then execute calbacks
                for callback in self._select_callbacks:
                    callback(self._current_shape_selection)
        except Exception as e:
            self.html.value = f"{str(e)}"

class JupyterSegmentationViewer:
    def __init__(self, file_stem, step_folder, seg_folder=None, logit_folder=None):
        self.file_stem = file_stem
        self.step_folder = step_folder
        assert step_folder.exists()
    
        solids, faces_dict = self.load_step()
        assert len(solids) == 1, "Expect only 1 solid"
        self.solid = solids[0]
        self.faces_dict = faces_dict
        self.entity_mapper = EntityMapper(self.solid.topods_shape())

        self.seg_folder = seg_folder
        self.logit_folder = logit_folder

        self.bit8_colors = [
            [235, 85, 79],  # ExtrudeSide
            [220, 198, 73], # ExtrudeEnd
            [113, 227, 76], # CutSide
            [0, 226, 124],  # CutEnd
            [23, 213, 221], # Fillet
            [92, 99, 222],  # Chamfer
            [176, 57, 223], # RevolveSide
            [238, 61, 178],  # RevolveEnd
            [176, 57, 100], # CutRevolveSide
            [138, 61, 178],  # CutRevolveEnd
        ]

        self.seg_dict = {"ExtrudeSide" : 0, 
            "ExtrudeEnd" : 1,
            "CutSide" : 2,
            "CutEnd" : 3,
            "Fillet" : 4,
            "Chamfer" : 5,
            "RevolveSide" : 6,
            "RevolveEnd" : 7,
            "CutExtrudeSide" : 2,
            "CutExtrudeEnd" : 3,
            "CutRevolveSide" : 8,
            "CutRevolveEnd" : 9,
        }

        self.color_map = ColorMap()

        self.selection_list = []

    def segment_color(self, seg_id):
        if seg_id in self.seg_dict.values():
            return self.format_color(self.bit8_colors[seg_id])
        else:
            return self.format_color([0, 0, 0])

    def format_color(self, c):
        return '#%02x%02x%02x' % (int(c[0]), int(c[1]), int(c[2]))

    def load_step(self):
        step_filename = self.step_folder / (self.file_stem + ".step")
        if not step_filename.exists():
            step_filename = self.step_folder / (self.file_stem + ".stp")
        assert step_filename.exists()
        return load_step_face_named(str(step_filename))

    def load_segmentation(self):
        """
        Load the seg file
        """
        assert not self.seg_folder is None,  "Must create this object specifying seg_folder"
        assert self.seg_folder.exists(), "The segmentation folder provided doesnt exist"

        seg_pathname = self.seg_folder / (self.file_stem + ".seg")
        return np.loadtxt(seg_pathname, dtype=np.uint64)

    def load_segmentation_cc3d(self):
        """
        Load the seg file from cc3d
        """
        from pandas import read_csv

        seg_pathname = self.seg_folder / (self.file_stem + ".seg.csv")
        data = read_csv(seg_pathname, header=0, index_col=0)

        segs = []
        for op in data["segment_type"]:
            if op in self.seg_dict:
                segs.append(self.seg_dict[op])
            else:
                segs.append(-1)
        return np.asarray(segs, dtype=np.uint64)

    def get_sw_face_index(self, face):
        """
        Convert a face color to an index. The index associates indexing of a face in SW api
        """
        face_name = self.faces_dict[face]
        decoded = int(face_name.replace("hbz", ""))
        return decoded

    def load_logits(self):
        """
        Load logits file
        """
        assert not self.logit_folder is None,  "Must create this object specifying logit_folder"
        assert self.logit_folder.exists(), "The logit folder provided doesnt exist"
        logit_pathname = self.logit_folder / (self.file_stem + ".logits")
        return np.loadtxt(logit_pathname)

    def select_face_callback(self, face):
        """
        Callback from the notebook when we select a face
        """
        face_index = self.entity_mapper.face_index(face)
        self.selection_list.append(face_index)

    def view_solid(self):
        """
        Just show the solid.  No need to show any segmentation data
        """
        renderer = MultiSelectJupyterRenderer()
        renderer.register_select_callback(self.select_face_callback)
        renderer.DisplayShape(
            self.solid.topods_shape(), 
            topo_level="Face", 
            render_edges=True, 
            update=True,
            quality=1.0
        )

    def view_colored_solid(self):
        self._display_faces_with_colors(self.faces_dict.keys(), self.faces_dict.values())

    def view_segmentation(self):
        """
        View the initial segmentation of this file
        """
        face_segmentation = self.load_segmentation()
        self._view_segmentation(face_segmentation)

    def view_segmentation_cc3d(self):
        """
        View the initial segmentation of this file
        """
        sw_face_segmentation = self.load_segmentation_cc3d()
        reindexing = [self.get_sw_face_index(face) for face in self.faces_dict.keys()]
        face_segmentation = sw_face_segmentation[reindexing]
        self._view_segmentation(face_segmentation)

    def view_predicted_segmentation(self):
        """
        View the segmentation predicted by the network
        """
        logits = self.load_logits()
        face_segmentation = np.argmax(logits, axis=1)
        self._view_segmentation(face_segmentation)

    def view_errors_in_segmentation(self):
        """
        View faces which are correct in green and incorrect in red
        """
        face_segmentation = self.load_segmentation()
        logits = self.load_logits()
        predicted_segmentation = np.argmax(logits, axis=1)
        correct_faces = (face_segmentation == predicted_segmentation)
        correct_color = self.format_color([0, 255, 0])
        incorrect_color = self.format_color([255, 0, 0])
        colors = []
        for prediction in correct_faces:
            if prediction:
                colors.append(correct_color)
            else:
                colors.append(incorrect_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def view_faces_for_segment(self, segment_index, threshold):
        logits = self.load_logits()
        logits_for_segment = logits[:,segment_index]
        faces_of_segment = logits_for_segment > threshold
        highlighted_color = self.format_color([0, 255, 0])
        other_color = self.format_color([156, 152, 143])
        colors = []
        for prediction in faces_of_segment:
            if prediction:
                colors.append(highlighted_color)
            else:
                colors.append(other_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    
    def highlight_faces_with_indices(self, indices):
        indices = set(indices)

        highlighted_color = self.format_color([0, 255, 0])
        other_color = self.format_color([156, 152, 143])

        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            if face_index in indices:
                colors.append(highlighted_color)
            else:
                colors.append(other_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def display_faces_with_heatmap(self, values, interval=None):
        if interval is None:
            norm_values = (values - np.min(values))/np.ptp(values)
        else:
            assert len(interval) == 2, "Interval must be length 1"
            interval_length = interval[1]-interval[0]
            assert interval_length > 0, "interval_length must be bigger than 0"
            norm_values = (values - interval[0])/(interval_length)
            norm_values = np.clip(norm_values, 0.0, 1.0)
        
        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            norm_value = norm_values[face_index]
            color_list = self.color_map.interpolate_color(norm_value)
            int_color_list = [int(v) for v in color_list]
            color = self.format_color(int_color_list)
            colors.append(color)

        self._display_faces_with_colors(self.solid.faces(), colors)


    def _view_segmentation(self, face_segmentation):
        colors = []
        print(f"num faces {self.solid.num_faces()}")
        for segment in face_segmentation:
            color = self.segment_color(segment)
            colors.append(color)
        self._display_faces_with_colors(self.solid.faces(), colors)


    def _display_faces_with_colors(self, faces, colors):
        """
        Display the solid with each face colored
        with the given color
        """
        renderer = JupyterRenderer()
        output = []
        for face, face_color in zip(faces, colors):
            result = renderer.AddShapeToScene(
                face.topods_shape(), 
                shape_color=face_color, 
                render_edges=True, 
                edge_color="#000000",
                quality=1.0
            )
            output.append(result)

        # Add the output data to the pickable objects or nothing get rendered
        for elem in output:
            renderer._displayed_pickable_objects.add(elem)                                         

        # Now display the scene
        renderer.Display()