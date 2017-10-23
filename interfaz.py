# -*- coding: utf-8 -*-	
import tkFont
import io
import sys
import os
import string
from Tkinter import *
from PIL import ImageTk, Image
from tkFileDialog import askopenfilename
import tkMessageBox as messagebox
import predecir_numero as pn
filePath = ""
pastfilePath = ""


#---------------------------------------------------------------
if __name__ == "__main__":

	root = Tk()
	root.resizable(0,0)
	root.title('Reconocedor de Numeros Manuscritos')
	w = 900
	h = 670
	x = 150
	y = 30

	respuesta =""
	root.geometry("%dx%d+%d+%d" % (w, h, x, y))
	
	times1 = tkFont.Font(family='Times',size=14, weight='bold')
	times2 = tkFont.Font(family='Times',size=12, weight='bold',slant = 'italic')
	times3 = tkFont.Font(family='Times',size=16, weight='bold',slant = 'italic')
	times3.configure(underline = True)


	la = Label(root, text="Ruta de Imagen: ", font = times2,background = 'white')
	la.place(x=10, y=30)

	bu = Button(root, text='Cargar Imagen',borderwidth = 1, command=updateImage,highlightbackground='black')
	bu.place(x=150, y=30)	
	
	pathText =Text(root, height=1, width=40)
	pathText.place(x=10, y=70)

	la = Label(root, text="Imagen:", font = times2,background = 'white')
	la.place(x=10, y=120)

	image = Image.open("imagenes/black.jpg")
	image = image.resize((5, h), Image.ANTIALIAS) # is (height, width)
	img = ImageTk.PhotoImage(image)
	li = Label(root, image=img,background = 'black')
	li.place(x=350, y=0)		
	#------------------
	la2 = Label(root, text="Imagen en escala de grises ►", font = times2,background = 'white')
	la2.place(x=390, y=50)

	la3 = Label(root, text="Imagen Desenfocada", font = times2,background = 'white')
	la3.place(x=390, y=165)
	la3 = Label(root, text="/Difuminada ►", font = times2,background = 'white')
	la3.place(x=420, y=185)

	la4 = Label(root, text="Imagen Binarizada", font = times2,background = 'white')
	la4.place(x=390, y=310)
	la41 = Label(root, text="Con Filtro Gaussiano ►", font = times2,background = 'white')
	la41.place(x=390, y=330)
	
	la5 = Label(root, text=u"Imagen Cerrada ►", font = times2,background = 'white')
	la5.place(x=390, y=460)

	la6 = Label(root, text="Digitos Identificados ►", font = times2,background = 'white')
	la6.place(x=390, y=595)
	#------------------
	bu = Button(root, text='Reconocer Numero',borderwidth = 5,highlightbackground='black', command=goQuery, font = times1)
	bu.place(x=70, y=450)

	

	root.protocol("WM_DELETE_WINDOW", on_closing)
	root.configure(background='white')
	root.mainloop()
