from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader




loader = PyPDFLoader('test_file.pdf')

doc = loader.load()



text = """
    Earth, the third planet from the Sun, is a uniquely vibrant celestial body teeming with life, water, and a rich variety of natural systems that have evolved over billions of years. Approximately 4.54 billion years old, Earth is the only known planet in the universe to support life, a fact attributed to its perfect distance from the Sun, its protective atmosphere, and the presence of liquid water, which covers about 71% of its surface. This blue planet has a slightly oblate spheroid shape due to its equatorial bulge, with a diameter of about 12,742 kilometers and a circumference of roughly 40,075 kilometers. Earth's structure is divided into several layers: the inner core, outer core, mantle, and crust. The inner core, composed primarily of iron and nickel, remains solid despite extreme temperatures due to immense pressure, while the outer core is liquid and responsible for generating Earth’s magnetic field. The mantle, a thick layer of semi-solid rock, facilitates tectonic movement, which leads to phenomena such as earthquakes, mountain formation, and volcanic activity. The crust, Earth’s outermost layer, is fragmented into tectonic plates that float atop the mantle, interacting in dynamic ways to shape the planet’s surface.

"""


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)


result = splitter.split_documents(doc)
print(result[0].page_content)