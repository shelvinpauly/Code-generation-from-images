import math, numpy as np

import os
import json
import itertools
from bs4 import BeautifulSoup
import re
import math, numpy as np
from zss import Node, simple_distance
from nltk.translate.bleu_score import sentence_bleu

# json_files = [pos_json for pos_json in os.listdir() if pos_json.endswith('.json')]


# def get_all_data(json_files):
#     data = []

#     for json_file in json_files:
#         with open(json_file) as f:
#             data.append(json.load(f))
#     return data



"""
  Given a dict representing a div, gives the left, top, width, height
  properties of said div

  Used for calculating intersections
"""
def unpack_div(div):
  return (div["left"], div["top"], div["width"], div["height"])





"""
  Given two divs that are circles, returns the area (pixels) of their intersection
"""
def circle_intersection(div1, div2):
    c1, c2 = unpack_div(div1), unpack_div(div2)
    c1_left, c1_top, c1_diameter, _= c1
    c2_left, c2_top, c2_diameter, _ = c2

    c1_center_x = c1_left + c1_diameter / 2
    c1_center_y = c1_top + c1_diameter / 2
    c2_center_x = c2_left + c2_diameter / 2
    c2_center_y = c2_top + c2_diameter / 2

    r = c1_diameter / 2
    R = c2_diameter / 2

    d = math.sqrt((c2_center_x - c1_center_x)**2 + (c2_center_y - c1_center_y)**2)

    # Check if there is an intersection
    if d < r + R:
        # Check if one circle is inside the other
        if d <= abs(r - R):
            # One circle is inside the other, intersection area is area of the smaller circle
            return math.pi * min(r, R)**2
        print(r, R, d)
        part1 = r**2 * math.acos((d**2 + r**2 - R**2) / (2 * d * r))
        part2 = R**2 * math.acos((d**2 + R**2 - r**2) / (2 * d * R))
        part3 = 0.5 * math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))

        return part1 + part2 - part3

    else:
        # No intersection
        return 0


def tokenize_code(code):
    return re.findall(r'\b\w+\b|\S', code)
  
  
"""
  Given two divs that are rectanges returns the area of their intersection
"""

def rectangle_intersection(div1, div2):
  
    r1 = unpack_div(div1)
    r2 = unpack_div(div2)

    r1_left, r1_top, r1_width, r1_height = r1
    r2_left, r2_top, r2_width, r2_height = r2

    r1_right = r1_left + r1_width
    r1_bottom = r1_top + r1_height

    r2_right = r2_left + r2_width
    r2_bottom = r2_top + r2_height

    # Check if there is an intersection
    if (r1_left < r2_right and r1_right > r2_left and
       r1_top < r2_bottom and r1_bottom > r2_top):
        # Calculate intersection rectangle
        inter_left = max(r1_left, r2_left)
        inter_top = max(r1_top, r2_top)
        inter_right = min(r1_right, r2_right)
        inter_bottom = min(r1_bottom, r2_bottom)
        # Calculate intersection area
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        return inter_area
    else:
        # No intersection
        return 0
    






'''
  Takes a div tag and returns a dict with the values:
  top: 25%
  left: 35%
  width: 19%
  height: 25%
  isCircle: False
  background-color: #FFEE22
'''
def render_div_attributes(buauSoupDivTag):
  tag = buauSoupDivTag
  style = tag.get('style')
  attributes = re.findall(r'([\w-]+)\s*:\s*([^;]+)', style)
  attributes_dict = {}
  for key, val in attributes:
    key = key.strip()
    val = val.strip()
    val = val.strip("%")
    if key != "border-radius" and key != "text-align" and key != "position" and key != "background-color":
      attributes_dict[key] = int(val)
    elif key == "position" or key == "background-color":
      attributes_dict[key] = val
    else:
      attributes_dict["isCircle"] = True
     
  if not "isCircle" in attributes_dict:
    attributes_dict["isCircle"] = False
  return attributes_dict


def convert_div_list(beauSoupDivs):
  res = []
  for div in beauSoupDivs:
    res.append(render_div_attributes(div))
  return res



"""
  Gets the area intersection of two divs
"""

def area_intersection(div1, div2):

  score = 0
  # assumes both divs are the same type
  if (div1["isCircle"] and not div2["isCircle"]) or (div2["isCircle"] and not div1["isCircle"]):
    score = 0
  if div1["isCircle"]:
    score = circle_intersection(div1, div2)
  else:
    score = rectangle_intersection(div1, div2)
  return score

def div_area(div):
  if (div["isCircle"]):
    return (div["width"]**2)*math.pi/4
  else:
    return div["width"]*div["height"]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove '#' if it exists
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Split and convert to decimal
    return r, g, b


def rgb_similarity(c1, c2):
    rgb1 = np.array(c1)
    rgb2 = np.array(c2)
    distance = np.linalg.norm(rgb1 - rgb2)
    max_distance = np.sqrt(3 * 255**2)
    similarity = 1 - distance / max_distance
    return similarity



def get_score(div1, div2):

  score = area_intersection(div1, div2)
  div1_area = div_area(div1)
  div2_area = div_area(div2)
  average = (div1_area + div2_area)/2
  score /= average
  color1 = hex_to_rgb(get_color(div1))
  color2 = hex_to_rgb(get_color(div2))
  color_similarity = rgb_similarity(color1, color2)


  return score*color_similarity


"""
  Gets the color of a div
"""
def get_color(div):
  return div["background-color"]

def CSSBLEU(truth_html, prediction_html):
  pred_soup = BeautifulSoup(prediction_html, 'html.parser')
  true_soup = BeautifulSoup(truth_html, 'html.parser')

  pred_divs = pred_soup.find_all("div")
  true_divs = true_soup.find_all("div")
  # print("passed")
  pred_divs_attributes = convert_div_list(pred_divs)
  true_divs_attributes = convert_div_list(true_divs)

  score = 0
  for pred_div in pred_divs_attributes:
    best_score = 0
    for true_div in true_divs_attributes:
      # print("getting score", true_div, pred_div)
      best_score = max(best_score, get_score(true_div, pred_div))
      # if best_score == 1:
      #   print("\n\n BEST FOR", pred_div, "is", true_div)
    score += best_score
  score /= len(pred_divs)
  return score








def html_to_tree(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup_to_tree(soup)
  
def tree_size(node):
    return 1 + sum(tree_size(child) for child in node.children)

def soup_to_tree(soup):
    node = Node(soup.name)
    for child in soup.children:
        if child.name is not None:
            node.addkid(soup_to_tree(child))
    return node

def tree_similarity(tree1, tree2):
    distance = simple_distance(tree1, tree2)
    size1 = tree_size(tree1)
    size2 = tree_size(tree2)
    dissimilarity = distance / (size1 + size2)
    similarity = 1 - dissimilarity
    return similarity

def DOMScore(html1, html2):
  tree1 = html_to_tree(html1)
  tree2 = html_to_tree(html2) 
  return tree_similarity(tree1, tree2)



def HTMLBLEU(refrenceHTML, candidateHTML, weights=[.33, 0, .33, .33]):
  Bleu_score = sentence_bleu([tokenize_code(refrenceHTML)], tokenize_code(candidateHTML))
  Weighted_BLEU = 0 # Needs to be implemented
  DOM_score = DOMScore(refrenceHTML, candidateHTML)
  CSS_score = CSSBLEU(refrenceHTML, candidateHTML)
  return (Bleu_score*weights[0]) +  (Weighted_BLEU*weights[1]) + (DOM_score*weights[2]) + (CSS_score*weights[2])
   
  
