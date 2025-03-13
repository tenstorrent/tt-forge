import lxml.etree
import pathlib

current_path = pathlib.Path(__file__).parent.joinpath('test.xml').resolve()
root = lxml.etree.parse(current_path)

# Select test case that do not contain the skipped element and fetch their properties elements.
#path = "/testsuites/testsuite/testcase[not(skipped)]/properties"
#path = /testsuites/testsuite/testcase[not(skipped)]/properties/property[@name="tags"]
path = "/testsuites/testsuite/testcase"
test_cases = root.xpath(path)

def fetch_all_model_names():
    a = root.xpath('/testsuites/testsuite/testcase/properties/property[@name="tags"]')
    json.loads(a[0].get('value'))
    print('a')

print()
fetch_all_model_names()

table = []


    
    


model_names = set()

for case in test_cases:
    if not case.xpath("skipped"): 
        for x in case:
            print(x)
     