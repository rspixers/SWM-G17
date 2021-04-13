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

    let amazon_model_metrics = ['MLP', 'Naive Bayes', 'Logistic Regression', 'SVM'];
    let amazon_accuracy = [0.6957083393139084,0.7398449834936127,0.7656093009903833,0.7645327974738051];
    let amazon_f1 = [0.5591599085048866,0.17029068436713207,0.4460651289009498,0.4169184290030212];

    let apple_model_metrics = ['SVM', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'MLP'];
    let apple_accuracy = [40,70,73,30,65]
    let apple_f1 = [62,43,76,73,51]

    function populate_barchart(data,accuracy,f1, id){
        var trace1 = {
            x: data,
            y: accuracy,
            name: 'Accuracy',
            type: 'bar',
            marker: {
                color: '#FFC627'
            }
        };

        var trace2 = {
            x: data,
            y: f1,
            name: 'F1 score',
            type: 'bar',
            marker: {
                color: '#8C1D40'
            }
        };

        var chart_data = [trace1, trace2];

        var layout = {
            barmode: 'group',
            title: 'Evaluation Metrics',
            xaxis: {
                title: "Models"
            },
            yaxis: {
                title: "Values"
            }
        };

        Plotly.newPlot(id, chart_data, layout);
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
    populate_barchart(amazon_model_metrics,amazon_accuracy, amazon_f1, 'amazon_results');
    populate_linechart(data, 'apple_momentum');
    populate_barchart(apple_model_metrics,apple_accuracy, apple_f1, 'apple_results');
    $("#aapl").addClass('hide')



    $(".submitform").on("click", function(ele){
        var url, text_value, selected_model, res_div;
        if($(ele.target).attr('id') == "ama_button"){
            url = "http://6c4fed3d59c5.ngrok.io/api/amazon";
            text_value = $("#testing_ama textarea").val();
            selected_model = $("#testing_ama select").val();
            res_div = $("#testing_ama .directions");
        }
        else{
            url = "http://6c4fed3d59c5.ngrok.io/api/apple";
            text_value = $("#testing_aapl textarea").val();
            selected_model = $("#testing_aapl select").val();
            res_div = $("#testing_aapl .directions");
        }

        var send_obj = {news_text: text_value, model: selected_model}
        console.log(send_obj);

        $.ajax({url: url, 
            data: send_obj ,
            dataType: 'json',
            success: function(result){
                console.log(result);
                $(res_div).html("");
                for (const key in result) {
                    $(res_div).append(result[key] ? "<div class='res positive'>"+ key +" predicted the news will INCREASE the stock price</div>"
                                                : "<div class='res negative'>"+ key +" predicted the news will DECREASE the stock price</div>" )
                }
            }
        });
    });

});







