import lxml.etree
import pathlib
import ast

current_path = pathlib.Path(__file__).parent.joinpath('test.xml').resolve()
root = lxml.etree.parse(current_path)

# Select test case that do not contain the skipped element and fetch their properties elements.
#path = "/testsuites/testsuite/testcase[not(skipped)]/properties"
#path = /testsuites/testsuite/testcase[not(skipped)]/properties/property[@name="tags"]
path = "/testsuites/testsuite/testcase"
test_cases = root.xpath(path)

def fetch_all_model_names():
    a = root.xpath('/testsuites/testsuite/testcase/properties/property[@name="tags"]')
    # string is a python dict not a json value due to single qoutes.
    c = { ast.literal_eval(x.get('value'))['model_name'] for x in a }
    return c 
    

def fetch_all_model_names():
    a = root.xpath('/testsuites/testsuite/testcase/properties/property[@name="tags"]')
    # string is a python dict not a json value due to single qoutes.
    c = { ast.literal_eval(x.get('value'))['model_name'] for x in a }
    return c 

a = fetch_all_model_names()


#['model','frontend','n150','n300']
    


model_names = set()

for case in test_cases:
    if not case.xpath("skipped"): 
        for x in case:
            print(x)
     