$(document).ready(function(){
    // the page ends loading

    // Make tables responsive
    $("table.docutils:not(.field-list,.footnote,.citation)")
        .wrap("<div class='wy-table-responsive'></div>");

    // Add extra class to responsive tables that contain
    // footnotes or citations so that we can target them for styling
    $("table.docutils.footnote")
        .wrap("<div class='wy-table-responsive footnote'></div>");
    $("table.docutils.citation")
        .wrap("<div class='wy-table-responsive citation'></div>");
});
