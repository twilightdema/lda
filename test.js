var lda = require('./lib/lda');

var text = 'Cats are small. Dogs are big. Cats like to chase mice. Dogs like to eat bones.';
var documents = text.match( /[^\.!\?]+[\.!\?]+/g );

var result = lda(documents, 2, 5);

console.log(JSON.stringify(result));
result.printReadableOutput();
