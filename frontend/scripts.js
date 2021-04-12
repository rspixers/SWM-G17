$( document ).ready(function() {

    // Amazon momentum
    var trace1 = {
        x: [1, 2, 3, 4],
        y: [10, 15, 13, 17],
        type: 'scatter'
    };
    
    var trace2 = {
        x: [1, 2, 3, 4],
        y: [16, 5, 11, 9],
        type: 'scatter'
    };
    
    var data = [trace1, trace2];


    function populate_linechart(data, id){
        Plotly.newPlot(id, data);
    }

    $("#amazon").on("click", function(){
        $("#aapl").addClass('hide');
        $("#ama").removeClass('hide');
        $("#ama_active").addClass('active_bar');
        $("#aapl_active").removeClass('active_bar');
    });

    $("#apple").on("click", function(){
        $("#aapl").removeClass('hide');
        $("#ama").addClass('hide');
        $("#ama_active").removeClass('active_bar');
        $("#aapl_active").addClass('active_bar');
    });

    populate_linechart(data, 'amazon_momentum');
    populate_linechart(data, 'amazon_results');
    populate_linechart(data, 'apple_momentum');
    populate_linechart(data, 'apple_results');
    $("#aapl").addClass('hide')



    $("#button").on("click", function(){
     console.log("Hello");
     var xmlhttp = new XMLHttpRequest();   // new HttpRequest instance 
     xmlhttp.open("POST", "http://0.0.0.0:8000/api/amazon");
     xmlhttp.setRequestHeader("Content-Type", "application/json");

     console.log('')   
     xmlhttp.onreadystatechange = function() {
        if (xmlhttp.readyState == XMLHttpRequest.DONE) {
            console.log(xmlhttp.responseText);
        }
    }


    xmlhttp.setRequestHeader("Access-Control-Allow-Origin", "*");
    xmlhttp.send(JSON.stringify({mytext:"nfreiuniv",model:"ofmo"}));




    

    });

});







