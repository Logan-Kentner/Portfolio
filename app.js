let viz;

// Tableau public dashboard link
const url = "https://public.tableau.com/views/PEACEUNSDGRevisited2025/PEACESDG2025?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link";

// DOM elements
const vizContainer = document.getElementById('vizContainer');
const launchVizBtn = document.getElementById('launchViz');
const exportPDF = document.getElementById('exportPDF');
const exportImage = document.getElementById('exportImage');

// Tableau options
const options = {
    hideTabs: true,
    height: 1000,
    width: 1200,
    onFirstInteraction: function() {
        console.log("Dashboard is interactive");
    }
};

// Init viz only on click
function initViz() {
    if (!viz) {
        viz = new tableau.Viz(vizContainer, url, options);
    }
}

// Event listener for Launch button
launchVizBtn.addEventListener("click", function (e) {
    e.preventDefault();
    initViz();
});

// PDF export
exportPDF.addEventListener("click", function () {
    if (viz) {
        viz.showExportPDFDialog();
    }
});

// Image export
exportImage.addEventListener("click", function () {
    if (viz) {
        viz.showExportImageDialog();
    }
});
