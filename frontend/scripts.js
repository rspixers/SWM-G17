$( document ).ready(function() {

    
    //  momentum
    function populate_momentum_linechart(){
        Plotly.d3.csv("../data/CHARTS/AMAZON1440.csv", function(err, rows){

            function unpack(rows, key) {
                return rows.map(function(row) { return row[key]; });
            }

            var trace1 = {
                type: "scatter",
                mode: "lines",
                name: 'AMZN',
                x: unpack(rows, 'Date'),
                y: unpack(rows, 'Open'),
                line: {color: '#FFD700'}
            }

            var data = [trace1];
            var layout = {
                title: 'Amazon Momentum',
                yaxis: {
                    title: "Values"
                }
            };
            Plotly.newPlot('amazon_momentum', data, layout);
        });

            Plotly.d3.csv("../data/CHARTS/APPLE1440.csv", function(err, rows){

                function unpack2(rows, key) {
                    return rows.map(function(row) { return row[key]; });
                }
        
                var trace2 = {
                    type: "scatter",
                    mode: "lines",
                    name: 'APPL',
                    x: unpack2(rows, 'Date'),
                    y: unpack2(rows, 'Open'),
                    line: {color: '#FFD700'}
                }

                var data2 = [trace2];
                var layout = {
                    title: 'Apple Momentum',
                    yaxis: {
                        title: "Values"
                    }
                };
                Plotly.newPlot('apple_momentum', data2, layout);
                $("#aapl").addClass('hide');
            });
            
        
    }

    let amazon_model_metrics = ['MLP', 'Naive Bayes', 'Logistic Regression', 'SVM','LSTM'];
    let amazon_accuracy = [0.6964977752260657,0.74228505813119,0.762092722836228,0.761446820726281, 0.974185908821788];
    let amazon_f1 = [0.562532326471501, 0.15525758645024704,0.43458980044345896, 0.39847991313789355, 0.9530374838431711];

    let apple_model_metrics =  ['MLP', 'Naive Bayes', 'Logistic Regression', 'SVM', 'LSTM'];
    let apple_accuracy = [ 0.8521125063266732,0.6529046459874832,0.8359965615032979,0.8365026953636532, 0.9653]
    let apple_f1 = [0.7419173933768892,0.6244893702087715,0.6978479026671798,0.6893214258453553, 0.9456]

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

            



    populate_momentum_linechart();

    // populate_barchart(amazon_model_metrics,amazon_accuracy, amazon_precision, amazon_recall, amazon_f1, 'amazon_results');
    populate_barchart(amazon_model_metrics,amazon_accuracy, amazon_f1, 'amazon_results');
    populate_barchart(apple_model_metrics,apple_accuracy, apple_f1, 'apple_results');




    $(".submitform").on("click", function(ele){
        var url, text_value, selected_model, res_div;
        $(".loader").removeClass('hide');
        let ngrok_url = "http://736ba434373f.ngrok.io/";
        if($(ele.target).attr('id') == "ama_button"){
            url = ngrok_url + "api/amazon";
            text_value = $("#testing_ama textarea").val();
            selected_model = $("#testing_ama select").val();
            res_div = $("#testing_ama .directions");
        }
        else{
            url = ngrok_url + "api/apple";
            text_value = $("#testing_aapl textarea").val();
            selected_model = $("#testing_aapl select").val();
            res_div = $("#testing_aapl .directions");
        }
        $(res_div).html("");

        var send_obj = {news_text: text_value, model: selected_model}
        console.log(send_obj);

        $.ajax({url: url, 
            data: send_obj ,
            dataType: 'json',
            success: function(result){
                console.log(result);
                $(".loader").addClass('hide');
                for (const key in result) {
                    $(res_div).append(result[key] ? "<div class='res positive'>"+ key +" predicted the news will INCREASE the stock price</div>"
                                                : "<div class='res negative'>"+ key +" predicted the news will DECREASE the stock price</div>" )
                }
            }
        });
    });

});







