var lda = require('./lib/lda');
var fs = require('fs');

//var text = 'Cats are small. Dogs are big. Cats like to chase mice. Dogs like to eat bones.';
//var documents = text.match( /[^\.!\?]+[\.!\?]+/g );

var doc_map = require('./coded_training_data.json');
var dict_map = require('./out_dict_map.json');
var rev_dict_map = require('./out_dict_rev_map.json');

var result = lda('map_provided',doc_map, 4, 10, ['th'], 0.1, 0.01, 100, dict_map, rev_dict_map);

console.log(JSON.stringify(result));
result.printReadableOutput();

fs.writeFileSync('model_output.json', JSON.stringify(result, null, 4));    

