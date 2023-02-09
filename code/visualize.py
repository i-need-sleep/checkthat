import argparse
import json

import dash
import pandas as pd
import Levenshtein

OUTPUT_DIR = '../results/outputs'
COLOR = {
    'text_highlight': 'red',
    'text_regular': 'black'
}

def visualise(args):
    app = dash.Dash(__name__)

    file_path = f'{OUTPUT_DIR}/{args.name}.json'

    with open(file_path, 'r') as f:
        data = json.load(f)

    content = [
        dash.html.Div(children=file_path),
    ]
    
    n_displayed = 0
    for line_idx, line in enumerate(data):

        if args.tuned:
            pred = line['tuned_pred']
            pred = pred == 1
        else:
            pred = line['pred']

        label = line['label']
        
        if args.failed_only and (label == 1 and pred) or (label == 0 and not pred):
            continue

        if line_idx < args.starting_idx:
            continue

        if n_displayed > args.max_n_lines + args.starting_idx:
            break

        text = line['text']
        img_path = f"assets/{line['img_path']}"

        if (label == 1 and not pred) or (label == 0 and pred):
            color = COLOR['text_highlight']
        else:
            color = COLOR['text_regular']
    
        content += [
            dash.html.Img(src = img_path),
            dash.html.Div(children=f'Label: {label}', style={'color': color}),
            dash.html.Div(children=f'Prediction: {pred}', style={'color': color}),
            dash.html.Div(children=f'Text input: {text}'),
            dash.html.P(children=['       ']),
        ]

        n_displayed += 1

    app.layout = dash.html.Div(children=content)

    app.run_server(debug=True)
    return
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='albef_mm_lr1e-4__tuned', type=str)

    parser.add_argument('--starting_idx', default='0', type=int)
    parser.add_argument('--max_n_lines', default='20', type=int)

    parser.add_argument('--tuned', action='store_true')
    parser.add_argument('--failed_only', action='store_true')

    args = parser.parse_args()

    visualise(args)