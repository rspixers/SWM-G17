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



    $(".submitform").on("click", function(ele){
        var url, text_value, selected_model, res_div;
        if($(ele.target).attr('id') == "ama_button"){
            url = "http://a0f6b00284fd.ngrok.io/api/amazon";
            text_value = $("#testing_ama textarea").val();
            selected_model = $("#testing_ama select").val();
            res_div = $("#testing_ama .directions");
        }
        else{
            url = "http://a0f6b00284fd.ngrok.io/api/apple";
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







