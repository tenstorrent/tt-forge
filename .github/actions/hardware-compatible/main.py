
import lxml.etree
import pathlib
import ast
from typing import List, Dict, Union

def get_property(root: lxml.etree.ElementTree, property_name: str) -> Dict[str, str]:
    a = root.xpath(f'property[@name="{property_name}"]')
    # string is a python dict not a json value due to single qoutes.
    value = a[0].get('value')
    if '{' in value:
        return ast.literal_eval(value)
    return value

pytest_files: List[str] = ['test_xla.xml', 'test_tt_forge-fe.xml', 'test_tt_torch.xml']

xml_parse_list: Dict[str, lxml.etree.ElementTree] = { x:  lxml.etree.parse(pathlib.Path(__file__).parent.joinpath(x).resolve()) for x in pytest_files }

# Select test case that do not contain the skipped element and fetch their properties elements.
#path = "/testsuites/testsuite/testcase[not(skipped)]/properties"
#path = /testsuites/testsuite/testcase[not(skipped)]/properties/property[@name="tags"]
path = "/testsuites/testsuite/testcase"

model_tests: Dict[str,List[Dict[str, str]]] = {}
#model_test['jax_mnist_cnn_nodropout_cv_image_cls_custom'] = [{'card': 'V100', 'status': 'PASSED', 'frontend': 'tt-xla', }]

for k, v in xml_parse_list.items():
    test_cases = v.xpath(path)
    for case in test_cases:
        if not case.xpath("skipped"): 
            for y in case:
                tag_attrs: Dict = get_property(y, 'tags')
                model_name = tag_attrs.get('model_name')
                if model_name is None:
                    continue
                temp_dict: Dict[str, str] = {}
                ## execution_phase only needed for tt-forge-fe other repos use bringup_status
                temp_dict['status'] = tag_attrs.get('bringup_status') if tag_attrs.get('bringup_status') else tag_attrs.get('execution_phase')
                temp_dict['card'] = 'n150'
                temp_dict['source_file'] = k
                temp_dict['frontend'] = get_property(y, 'owner')
                if model_tests.get(model_name):
                    model_tests[model_name].append(temp_dict)
                    continue
                model_tests[tag_attrs.get('model_name')] = [temp_dict]
    
    
print(model_tests) 
print('a')