var lda = require('./lib/lda');
var perplexity_lda = require('./lib/perplexity_lda');

var text = 'Cats are small. Dogs are big. Cats like to chase mice. Dogs like to eat bones.';
var text_test = 'Cats are small. You are small. Tigers are big, Cats chase mice. Dogs eat mice.';
var documents = text.match( /[^\.!\?]+[\.!\?]+/g );
var documents_test = text_test.match( /[^\.!\?]+[\.!\?]+/g );

var result = lda('direct', documents, 2, 5, ['th']);

console.log(JSON.stringify(result));
result.printReadableOutput();

var perplexity = perplexity_lda(result.topicModel, documents_test, ['th']);
console.log('Perplexity = '+perplexity);
