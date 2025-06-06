console.log('Is this working?');

let viz;

//Add Share Link to Tableau Public in here
const url = "https://public.tableau.com/views/PEACEUNSDGRevisited2025/PEACESDG2025?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link";

const vizContainer = document.getElementById('vizContainer');
const options = {
    hideTabs: true,
    height: 1000,
    width: 1200,
    onFirstInteraction: function() {
        workbook = viz.getWorkbook();
        activeSheet = workbook.getActiveSheet();
        console.log("My dashboard is interactive");
    }
};

//create a function to generate the viz element
function initViz() {
    console.log('Executing the initViz function!');
    viz = new tableau.Viz(vizContainer, url, options);
}

// run the initViz function when the page loads
launchVizLink.addEventListener("click", function (e) {
    e.preventDefault(); // prevent default hyperlink behavior
    initViz();
});

//click on the pdf button to generate pdf of dashboard
function generatePDF() {
    viz.showExportPDFDialog()
}

exportPDF.addEventListener("click", function () {
    generatePDF();
  });

//click on image to generate image of dashboard
function generateImage() {
    viz.showExportImageDialog()
}

exportImage.addEventListener("click", function () {
    generateImage();
  });
