import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm 
import math
import yaml

class MainFrame(ttk.Frame):
    def __init__(self, mainframe, parameters):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Bounding Box Annotation Tool')

        self.column2_frame = ttk.Frame(self.master)
        self.list_frame = ListFrame(self.column2_frame, row=1, col=0)

        self.image_frame = ImageFrame(self.master, parameters=parameters, list_frame=self.list_frame)

        self.panel_frame = PanelFrame(self.column2_frame, image_frame=self.image_frame, list_frame=self.list_frame, row=0, col=0)
        self.column2_frame.grid(row=0, column=2)

        self.window_height = self.image_frame.image.height + 50
        self.window_width = self.image_frame.image.width + 350
        self.master.geometry(f"{self.window_width}x{self.window_height}")
        self.master.state("normal")

        self.master.rowconfigure(0, weight=1)  
        self.master.columnconfigure(0, weight=1)

class AutoScrollbar(ttk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class ImageFrame(ttk.Frame):
    def __init__(self, mainframe, parameters, list_frame):
        self.master = mainframe
        self.params = parameters
        self.list_frame = list_frame
    
        vbar = AutoScrollbar(self.master, orient='vertical') 
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=2, column=0, sticky='we')
        
        self.max_image_width = 1920
        self.max_image_height = 1080
        
        self.canvas = tk.Canvas(self.master, highlightthickness=10, xscrollcommand=hbar.set, yscrollcommand=vbar.set)  
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()   

        vbar.configure(command=self.scroll_y)  
        hbar.configure(command=self.scroll_x)

        self.canvas.bind('<Configure>', self.show_image)  
        self.canvas.bind('<MouseWheel>', self.wheel)   
        self.canvas.bind('<Button-5>',   self.wheel)  
        self.canvas.bind('<Button-4>',   self.wheel)  
        self.canvas.bind('<ButtonPress-2>', self.move_from)
        self.canvas.bind("<B2-Motion>", self.move_to)
        self.canvas.bind("<Motion>", self.mouse_move)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move_press)
        self.rect = None
        self.labels = []
        self.start_x = None
        self.start_y = None
        self.x = self.y = 0
        self.list_figures = []

        self.canvas.bind("<Button-3>", self.on_right_mouse_button_press)

        # image
        self.classes_names = ["--"]
        self.image =  Image.open(os.path.join(os.path.abspath(os.getcwd()), "figures/empty.jpg"))
        self.width, self.height = self.image.size

        # resize image if too big
        if self.width > (0.8 * self.max_image_width) or self.height > (0.8 * self.max_image_height):
            print("reizing image from {}x{}".format(self.width, self.height))

            new_width = 0.8 * self.max_image_width
            new_height = int((float(self.height)*float((new_width/float(self.width)))))

            self.image = self.image.resize((int(new_width), int(new_height)))
            self.width, self.height = self.image.size

        self.imscale = 1.0  
        self.delta = 1.2
        #self.container = self.canvas.create_rectangle(0.5*(self.master.winfo_width() - self.width), 0.5*(self.master.winfo_height() - self.height), 0.5*(self.master.winfo_width() + self.width), 0.5*(self.master.winfo_height() + self.height), width=0)
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0) 
        self.show_image()

        self.bbox_container = None
        self.bbox_image = None
        self.delta_moveto = (0,0)

        # display mouse position
        self.ctrPanel = tk.Frame(self.master)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = tk.W+tk.E)
        self.disp = tk.Label(self.ctrPanel, text='')
        self.disp.pack(side = tk.RIGHT)


    def scroll_y(self, *args, **kwargs):
        self.canvas.yview(*args, **kwargs)   
        self.show_image()  


    def scroll_x(self, *args, **kwargs):
        self.canvas.xview(*args, **kwargs)  
        self.show_image()


    def move_from(self, event):
        self.canvas.scan_mark(event.x, event.y)


    def move_to(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.delta_moveto = (-self.canvas.canvasx(0), -self.canvas.canvasy(0))
        self.show_image()  


    def on_mouse_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')


    def on_right_mouse_button_press(self, event):
        labels = self.addBB()
        if labels is not None:
            self.list_frame.addLabel(labels)


    def on_mouse_move_press(self, event):
        self.curX = self.canvas.canvasx(event.x)
        self.curY = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)    


    def wheel(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # get image area
        bbox = self.canvas.bbox(self.container)  

        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area

        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            if self.imscale <= 1: return # img all zoomed out!
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale        *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()


    def show_image(self, event=None):

        # image area
        bbox1 = self.canvas.bbox(self.container) 
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1) # Remove 1 pixel shift at the sides of the bbox1

        # visible area canvas
        bbox2 = (self.canvas.canvasx(0), 
                self.canvas.canvasy(0),
                self.canvas.canvasx(self.canvas.winfo_width()),
                self.canvas.canvasy(self.canvas.winfo_height()))

        # scroll region box
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]

        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]

        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        
        bbox = [int(b) for b in bbox] # fix issue in Ubuntu20

        # set scroll region
        self.canvas.configure(scrollregion=bbox)  

        # get coordinates (x1,y1,x2,y2) of the image tile
        x1 = max(bbox2[0] - bbox1[0], 0)  
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        self.bbox_container = bbox1
        self.bbox_image = (x1, y1, x2, y2)

        # show image if it in the visible area
        if int(x2 - x1) > 0 and int(y2 - y1) > 0: 
            x = min(int(x2 / self.imscale), self.width)   
            y = min(int(y2 / self.imscale), self.height)   
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                            anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  
            self.canvas.imagetk = imagetk 


    def update_image(self, path):

        self.DATA_FOLDER = path
        self.imgs_dir = os.path.join(self.DATA_FOLDER, self.params.images_folder)
        self.output_dir = os.path.join(self.DATA_FOLDER, self.params.labels_folder)

        self.img_files = self.get_imgs_inside_folder()
        self.img_counter_step = 100 / (len(self.img_files)-1)
        self.img_counter = 0

        # classes
        self.classes_names = self.get_classes_from_file()
        self.idx = 0

        # colormap
        cmap = cm.get_cmap('tab10', len(self.classes_names))    
        self.classes_colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]
        self.classes_colors_rgb = [cmap(i) for i in range(cmap.N)]

        self.image = Image.open(os.path.join(self.imgs_dir, self.img_files[0]))
        self.width, self.height = self.image.size

        # resize image if too big
        if self.width > (0.8 * self.max_image_width) or self.height > (0.8 * self.max_image_height):
            print("reizing image from {}x{}".format(self.width, self.height))

            new_width = 0.8 * self.max_image_width
            new_height = int((float(self.height)*float((new_width/float(self.width)))))

            self.image = self.image.resize((int(new_width), int(new_height)))
            self.width, self.height = self.image.size

        self.imscale = 1.0  
        self.delta = 1.2
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0) 
        self.show_image()


    def mouse_move(self, event):
        self.disp.config(text = 'mouse pos = x: %d, y: %d' %(event.x, event.y))


    def get_imgs_inside_folder(self):
        img_paths = [f for f in os.listdir(self.imgs_dir) if not f.startswith('.')]
        img_paths.sort()
        return img_paths


    def get_classes_from_file(self):
        fp = open(os.path.join(self.DATA_FOLDER, self.params.classes_file), "r")
        classes_names = fp.read().split("\n")[:-1]
        return classes_names


    def showBBoxes(self):
        def rescale(l, width, height):
            x,y,w,h = l

            cx = (x - w/2) * width
            cy = (y - h/2) * height
            ww = w * width
            hh = h * height

            #print("rescale: ", cx / self.imscale, cy / self.imscale, ww / self.imscale, hh / self.imscale)
            return cx / self.imscale, cy / self.imscale, ww / self.imscale, hh / self.imscale

        # current img 
        current_img_file = self.img_files[self.img_counter]

        # save labels 
        name_txt = current_img_file.split(".")[0] + ".txt"
        label_txt_filename = os.path.join(self.output_dir, name_txt)
        

        if os.path.exists(label_txt_filename):
            labels = np.loadtxt(label_txt_filename)
        else:
            print(f"txt does not exist! {label_txt_filename}")
            return

        if labels.size == 0:
            print(f"empty txt: {label_txt_filename}")
            return

        width = self.bbox_container[2] - self.bbox_container[0] 
        height = self.bbox_container[3] - self.bbox_container[1] 
        #print("width, height ", width, height)

        if labels.ndim == 1:
            x, y, w, h = rescale(labels[1:], width, height)
            color = self.classes_colors_rgb[int(labels[0])] 
            draw = ImageDraw.Draw(self.image)
            draw.rectangle([(x,y),(x+w,y+h)], outline=(int(color[0]*255), int(color[1]*255), int(color[2]*255)), width=2)

        else:
            for label in labels:
                x, y, w, h = rescale(label[1:], width, height)
                color = self.classes_colors_rgb[int(label[0])]   
                draw = ImageDraw.Draw(self.image)
                draw.rectangle([(x,y),(x+w,y+h)], outline=(int(color[0]*255), int(color[1]*255), int(color[2]*255)), width=2)

        self.show_image()


    def transform(self, x, y):
        x_world = x + self.delta_moveto[0]
        y_world = y + self.delta_moveto[1]
        return x_world, y_world        


    def computeBBox(self):
        
        x1, y1 = self.transform(self.start_x, self.start_y)
        x2, y2 = self.transform(self.curX, self.curY)
        x0, y0 = self.transform(self.bbox_container[0], self.bbox_container[1])

        x = min(x1, x2) - x0
        y = min(y1, y2) - y0
        w = math.fabs(x1 - x2)
        h = math.fabs(y1 - y2)

        width = self.bbox_container[2] - self.bbox_container[0] 
        height = self.bbox_container[3] - self.bbox_container[1] 

        max_x = max(x, 0)
        max_y = max(y, 0)
        max_x = min(max_x, width)
        max_y = min(max_y, height)

        w = float(w) / float(width)
        h = float(h) / float(height)
        x = float(max_x) / float(width)
        y = float(max_y) / float(height)
        return float(x+w/2.), float(y+h/2.), w, h


    def addBB(self):
        if self.rect:
            # bbox
            cx,cy,w,h = self.computeBBox()
            self.labels.append([self.idx, cx, cy, w, h])

            self.canvas.delete(self.rect)
            self.rect = None

            # visualization
            rect = self.canvas.create_rectangle(self.curX, self.curY, self.start_x, self.start_y, outline=self.classes_colors[self.idx], width=2)
            self.list_figures.append(rect)
            print(f"ADD: {self.idx, cx, cy, w, h}")
            return self.labels
        else:
            return None


    def next(self):
        if self.img_counter >= len(self.img_files):
            print("end reached!")
            return

        # current img 
        current_img_file = self.img_files[self.img_counter]

        # save labels 
        name_txt = current_img_file.split(".")[0] + ".txt"
        label_txt_filename = os.path.join(self.output_dir, name_txt)


        if self.labels:
            f = open(label_txt_filename, "a")
            np.savetxt(f, self.labels, fmt='%d %1.4f %1.4f %1.4f %1.4f')       
            f.close()    

        self.clear()
        self.img_counter +=1
        if self.img_counter >= len(self.img_files):
            print("end reached!")
            return

        # new img 
        current_img_file = self.img_files[self.img_counter]

        self.image = Image.open(os.path.join(self.imgs_dir, current_img_file))
        
        self.show_image()

        print(f"SAVED: {label_txt_filename}, NEXT IMG: {current_img_file}")


    def back(self):

        if self.img_counter == 0:
            print("beginning reached!")
            return

        self.clear()      
        self.img_counter -=1

        # new img 
        current_img_file = self.img_files[self.img_counter]  
        self.image = Image.open(os.path.join(self.imgs_dir, current_img_file))
        self.show_image()

        print(f"PREVIOUS IMG: {current_img_file}") 


    def clear_figures(self):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        for fig in self.list_figures:
            self.canvas.delete(fig)
        self.list_figures = []

        self.start_x = None
        self.start_y = None
        self.x = self.y = 0

    def clear(self, btn=False):
        self.clear_figures()
        self.labels = []

        if btn:
            self.image = Image.open(os.path.join(self.imgs_dir, self.img_files[self.img_counter]))
            self.show_image()


    def deleteFiguresInOrder(self):
        if self.rect: # current line
            self.canvas.delete(self.rect)
            self.rect = None
        elif len(self.list_figures) > 0:
            self.canvas.delete(self.list_figures[-1])
            self.list_figures.pop(-1)
            self.labels.pop(-1)

    def deleteLabel(self):
        
        current_img_file = self.img_files[self.img_counter]
        label_txt_path = os.path.join(self.output_dir, current_img_file.split(".")[0] + ".txt")

        if os.path.exists(label_txt_path):
            os.remove(label_txt_path)

        print("deleted: ", label_txt_path)

        self.image = Image.open(os.path.join(self.imgs_dir, self.img_files[self.img_counter]))
        self.show_image()


class ChooseImage(ttk.Frame):
    def __init__(self, mainframe, image_frame, input_frame):
        
        self.image_frame = image_frame
        self.master = mainframe
        self.input_frame = input_frame

        self.selection_frame = ttk.Frame(master=self.master)

        self.input_box = ttk.Frame(master=self.selection_frame)
        self.title_box = tk.Label(master=self.input_box, text="Input:", font='bold', width=8, height=1)
        self.title_box.pack(side=tk.LEFT, padx=5, pady=5) 

        self.btn_choose = tk.Button(master=self.input_box, font='bold', text = "choose data dir", width = 18, height = 1, command = self.choose_dir)
        self.btn_choose.pack(side=tk.LEFT, padx=5, pady=5)
        self.input_box.grid(row=0, column=0, sticky='nesw')


        self.source_dir = ""
        self.txt_box = tk.Text(master=self.selection_frame, font='bold', width=30, height=1)
        self.txt_box.grid(row=3, column=0, sticky="nesw", padx=5, pady=5)
        self.txt_box.insert(tk.END,"no directory selected!") 

        self.selection_frame.grid(row=2, column=0, sticky='nesw')



    def choose_dir(self):
        self.source_dir = filedialog.askdirectory(parent=self.master, initialdir= os.getcwd(), title='please select a directory')
        self.txt_box.delete(1.0,tk.END)
        self.txt_box.insert(tk.END,self.source_dir) 
        self.image_frame.update_image(self.source_dir)
        self.input_frame.update()


class InputFrame(ttk.Frame):
    def __init__(self, mainframe, imgframe, listframe):
        self.master = mainframe
        self.imgframe = imgframe
        self.listframe = listframe

        ## INPUTS FRAME -----------------------
        self.inputs_frame = ttk.Frame(self.master)

        self.title_box = tk.Label(master=self.inputs_frame, text="Control Panel", font='bold', width=14, height=1)
        self.title_box.grid(row=0, column=0,  padx=5, pady=5)

        self.sep = ttk.Separator(master=self.inputs_frame, orient=tk.HORIZONTAL)
        self.sep.grid(row=1, column=0, padx=5, pady=5, sticky='nesw')

        # buttons arrows
        self.frame_arrows = ttk.Frame(self.inputs_frame)
        self.btn_3 = tk.Button(master=self.frame_arrows, text="<<", font='bold', width=12, height=1, command = self.on_previous_press)
        self.btn_3.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_3 = tk.Button(master=self.frame_arrows, text=">>", font='bold', width=12, height=1, command = self.on_next_press)
        self.btn_3.pack(side=tk.LEFT, padx=5, pady=5)
        self.frame_arrows.grid(row=3, column=0, sticky='s')

        self.frame_bar = ttk.Frame(self.inputs_frame)
        self.bar_style = ttk.Style(self.frame_bar)
        self.bar_style.layout("LabeledProgressbar",  # add the label to the progressbar style
                [('LabeledProgressbar.trough',
                {'children': [  ('LabeledProgressbar.pbar', {'side': 'left', 'sticky': 'ns'}),
                                ("LabeledProgressbar.label", {"sticky": ""}),
                             ],
                'sticky': 'nswe'})])
        self.progress=ttk.Progressbar(self.frame_bar, orient=tk.HORIZONTAL, length=300, mode='determinate', style="LabeledProgressbar")
        self.progress.pack(ipady=5, padx=5, pady=5)
        self.bar_style.configure("LabeledProgressbar", text="-- / --      ", background="#0088CC")
        self.frame_bar.grid(row=2, column=0, sticky='nesw')


        # buttons clear 
        self.frame_clear = ttk.Frame(self.inputs_frame)
        self.btn_clear = tk.Button(master=self.frame_clear, text="clear all", font='bold', width=12, height=1, command = self.on_button_clear_press)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_clear = tk.Button(master=self.frame_clear, text="clear last", font='bold', width=12, height=1, command = self.on_button_clear_last_press)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5)
        self.frame_clear.grid(row=4, column=0, sticky='nesw')

        ###
        self.btn_add = tk.Button(master=self.inputs_frame, text="add bounding box", width=28, font='bold', height=1, command = self.on_button_add_press)
        self.btn_add.grid(row=5, column=0, padx=5, pady=5)


        ##### drodown menu
        self.frame_labels = ttk.Frame(self.inputs_frame)

        self.txt_box = tk.Label(master=self.frame_labels, text="class active:", font='bold', width=14, height=1)
        self.txt_box.pack(side=tk.LEFT,  padx=(5,8), pady=5)

        self.variable_class = tk.StringVar(self.frame_labels)
        self.variable_class.set(self.imgframe.classes_names[0]) # default value
        self.variable_class.trace("w", self.on_menu_change)

        self.imgframe.idx = 0
        self.opt_menu = ttk.Combobox(self.frame_labels, textvariable=self.variable_class, state='readonly', width=13, height=10, font='bold') #, *self.imgframe.classes_names)
        self.opt_menu.pack(side=tk.LEFT, padx=5, pady=5)
        self.frame_labels.grid(row=6, column=0, sticky='nesw')


        self.sep = ttk.Separator(master=self.inputs_frame, orient=tk.HORIZONTAL)
        self.sep.grid(row=7, column=0, padx=5, pady=5, sticky='nesw')


        ## LABELS TXT BUTTONS
        self.frame_labels_txt = ttk.Frame(self.inputs_frame)

        self.title_box = tk.Label(master=self.frame_labels_txt, text="Txt Labels", font='bold', width=14, height=1)
        self.title_box.grid(row=0, column=0,  padx=5, pady=5)

        self.frame_txt = ttk.Frame(self.frame_labels_txt)
        self.btn_show = tk.Button(master=self.frame_txt, text="show", font='bold', width=12, height=1, command = self.on_show_press)
        self.btn_show.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_delete_labels = tk.Button(master=self.frame_txt, text="delete file", font='bold', width=12, height=1, command = self.on_button_delete_labels_press)
        self.btn_delete_labels.pack(side=tk.LEFT, padx=5, pady=5)
        self.frame_txt.grid(row=1, column=0, sticky='nesw')

        self.frame_labels_txt.grid(row=8, column=0, sticky='nesw')


        self.inputs_frame.grid(row=0, column=0, sticky='nesw')


    def on_menu_change(self, *args):
        cls = self.variable_class.get()
        self.imgframe.idx = self.imgframe.classes_names.index(cls)  # id label 
        print(f"CHANGED CLASS: {cls} -- {self.imgframe.idx}")

    def on_button_clear_press(self):
        self.imgframe.clear(btn=True)

    def on_button_clear_last_press(self):
        self.imgframe.deleteFiguresInOrder()
        self.listframe.deleteLabel()

    def on_button_add_press(self):
        labels = self.imgframe.addBB()
        if labels is not None:
            self.listframe.addLabel(labels)

    def on_next_press(self):
        self.imgframe.next() 
        if self.progress['value'] < 100 - self.imgframe.img_counter_step:
            self.progress['value']+=self.imgframe.img_counter_step
            self.bar_style.configure("LabeledProgressbar", text="{0} / {1}      ".format(self.imgframe.img_counter+1, len(self.imgframe.img_files)))
            self.frame_bar.update_idletasks()

        self.listframe.clearList()

    def on_previous_press(self):
        self.imgframe.back()
        if self.progress['value'] > 0:
            self.progress['value']-=self.imgframe.img_counter_step
            self.bar_style.configure("LabeledProgressbar", text="{0} / {1}      ".format(self.imgframe.img_counter+1, len(self.imgframe.img_files)))
            self.frame_bar.update_idletasks()

        self.listframe.clearList()


    def on_show_press(self):
        self.imgframe.showBBoxes()
    
    def on_button_delete_labels_press(self):
        self.imgframe.deleteLabel()

    def update(self):
        '''
        self.opt_menu['menu'].delete(0, 'end')
        for name in self.imgframe.classes_names:
            self.opt_menu['menu'].add_command(label=name, command=lambda name=name: self.variable_class.set(name))
        self.variable_class.set(self.imgframe.classes_names[0])
        '''
        self.opt_menu['values'] = self.imgframe.classes_names
        self.variable_class.set(self.imgframe.classes_names[0])

        self.bar_style.configure("LabeledProgressbar", text="{0} / {1}      ".format(self.imgframe.img_counter, len(self.imgframe.img_files)))
        self.frame_bar.update_idletasks()



class ListFrame(ttk.Frame):
    def __init__(self, mainframe, row=1, col=2):

        self.list_frame = tk.Frame(mainframe)

        self.title_box = tk.Label(master=self.list_frame, text="Bounding Boxes:", font='bold', width=14, height=1)
        self.title_box.grid(row=0, column=0,  padx=5, pady=5)

        self.listbox = tk.Listbox(master=self.list_frame, width = 32, height = 100)
        self.listbox.grid(row=1, column=0)

        self.list_frame.grid(row=row, column=col)


    def addLabel(self, labels):
        last_label = ["%.2f"%l for l in labels[-1]]
        self.listbox.insert(0, last_label) 

    def deleteLabel(self):
        self.listbox.delete(0)

    def clearList(self):
        self.listbox.delete(0,'end')


class PanelFrame(ttk.Frame):
    def __init__(self, mainframe, image_frame, list_frame, row=0, col=2):

        self.panel_frame = ttk.Frame(mainframe)
        self.input_frame = InputFrame(mainframe=self.panel_frame, imgframe=image_frame, listframe=list_frame)

        self.sep = ttk.Separator(master=self.panel_frame, orient=tk.HORIZONTAL)
        self.sep.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.choose_frame = ChooseImage(mainframe=self.panel_frame, image_frame=image_frame, input_frame=self.input_frame)

        self.sep = ttk.Separator(master=self.panel_frame, orient=tk.HORIZONTAL)
        self.sep.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

        self.panel_frame.grid(row=row, column=col)


class Params():
    def __init__(self, params_file):
        values = None
        with open(params_file, 'r') as stream:
            try:
                values = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.images_folder = values["images_folder"]
        self.labels_folder = values["labels_folder"]
        self.classes_file = values["classes_file"]


if __name__ == "__main__":
    params = Params("params.yaml")           
    root = tk.Tk()
    app = MainFrame(root, parameters=params)
    root.mainloop()
