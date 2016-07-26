var http = require("http");
var fs = require('fs');
var url = require('url');
var PythonShell = require('python-shell');

var options = {
    mode: 'text'
}

var pyshell = new PythonShell('transferFromServer.py', options);

pyshell.on('message', function (message) {
    var data = message.split(" ");
    console.log(data);
});

/*
PythonShell.run('imgprocess.py', function (err) {
    if (err) throw err;
    console.log('finished');
});
*/

http.createServer(function (request, response) {

    var parsed = url.parse(request.url);
    if (request.method == 'GET') {
        //console.log("GET request recieved");
        if (parsed.pathname == '/') {
            fs.readFile('./index.html', function(err, file) {
                if(err) {
                    return console.log(err);
                }
            response.writeHead(200, { 'Content-Type': 'text/html' });
            response.end(file, "utf-8");
            });
        }
    }
    if (request.method == 'POST') {
        //console.log("POST request received");
        body = "";
        request.on('data', function (data) {
            body += data;
            if (body.length > 1e6)
                request.connection.destroy();
        });
        request.on('end', function () {
            pyshell.send(body);
        });
        response.writeHead(200, {"Content-Type": "application/json"});
        response.write(JSON.stringify(tosend));
        response.end("");
    }

}).listen(8000);

console.log('Server running at http://127.0.0.1:8000/');

