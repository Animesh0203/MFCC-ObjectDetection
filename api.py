import json
import datamuse

def apply_template(input_file, template_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    with open(template_file, 'r', encoding='utf-8') as template_file:
        template_data = json.load(template_file)

    merged_data = {**template_data, **json_data}

    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, indent=4)

def merge_json(template, data):
    if isinstance(template, dict) and isinstance(data, dict):
        merged = {}

        for key in template:
            if key in data:
                merged[key] = merge_json(template[key], data[key])
            else:
                merged[key] = merge_json(template[key], None)

        for key in data:
            if key not in template:
                merged[key] = data[key]

        return merged
    else:
        return template if data is None else data

# Usage
apply_template('input.json', 'template.json', 'output.json')
