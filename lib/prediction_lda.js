var stem = require('stem-porter');

var process = function(model, sentences, languages, alphaValue, randomSeed) {

  var V = model.hypers.W;
  var K = model.hypers.T;
  var vocab = model.hypers.vocab;
  var vocabOrig = model.hypers.vocabOrig;

  var _alpha = model.priors.alpha;
  var beta = model.priors.beta;

  var theta = model.posteriors.theta;
  var phi = model.posteriors.phi;
  
  var nw = model.counters.nw;
  var nwsum = model.counters.nwsum;

  // Result is perplexity of the model
  var result = 0;

  // Index-encoded array of sentences, with each row containing the indices of the words in the vocabulary.
  var documents = new Array();
  // Hash of vocabulary words and the count of how many times each word has been seen.
  var f = {};
  // Vocabulary of unique words in their original form.
  for(var i=0;i<vocab.length;i++) {
    f[vocab[i]] = 1;
  }
  // Array of stop words
  languages = languages || Array('en');

  if (sentences && sentences.length > 0) {
    var stopwords = new Array();

    languages.forEach(function(value) {
        var stopwordsLang = require('./stopwords_' + value + ".js");
        stopwords = stopwords.concat(stopwordsLang.stop_words);
    });

    for(var i=0;i<sentences.length;i++) {
      if (sentences[i]=="") continue;
      documents[i] = new Array();

      var words = sentences[i] ? sentences[i].split(/[\s,\"]+/) : null;

      if(!words) continue;
      for(var wc=0;wc<words.length;wc++) {
        var w=words[wc].toLowerCase();
        if(languages.indexOf('en') != -1)
          w=w.replace(/[^a-z\'A-Z0-9\u00C0-\u00ff ]+/g, '');
        var wStemmed = stem(w);
        if (w=="" || !wStemmed || w.length==1 || stopwords.indexOf(w.replace("'", "")) > -1 || stopwords.indexOf(wStemmed) > -1 || w.indexOf("http")==0) continue;
        if (f[wStemmed]) { 
            f[wStemmed]=f[wStemmed]+1;
            documents[i].push(vocab.indexOf(wStemmed));
          } 
        else if(wStemmed) { 
          // We use -1 to indicate verbatim that is not existing in our model dictionary.
          // documents[i].push(-1);
        };            
      }
    }

    var M = documents.length;
    var alpha = alphaValue || _alpha;  // per-document distributions over topics
    documents = documents.filter((doc) => { return doc.length }); // filter empty documents

    console.log('docs length = '+documents.length);


    console.log('Start Prediction...');
    lda_predictnext.configure(documents,vocab,V, 10, 2000, 100, 10, randomSeed);
    console.log('Start running left-to-right algorithm...');
    result = lda_predictnext.predict_next_word(K, alpha, beta,
      nw, nwsum
    );
  }
  return result;
}

function makeArray(x) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=0;
    }
    return a;
}

function make2DArray(x,y) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=new Array();
        for (var j=0;j<y;j++)
            a[i][j]=0;
    }
    return a;
}

var lda_predictnext = new function() {
    var documents,nw,nwsum,V,K,alpha,beta; 
    var THIN_INTERVAL = 20;
    var BURN_IN = 100;
    var ITERATIONS = 1000;
    var SAMPLE_LAG;
    var RANDOM_SEED;
    var dispcol = 0;
    var numstats=0;
    var vocab = [];
    this.configure = function (docs,vocab,v,iterations,burnIn,thinInterval,sampleLag,randomSeed
    ) {
        this.ITERATIONS = iterations;
        this.BURN_IN = burnIn;
        this.THIN_INTERVAL = thinInterval;
        this.SAMPLE_LAG = sampleLag;
        this.RANDOM_SEED = randomSeed;
        this.documents = docs;
        this.V = v;
        this.dispcol=0;
        this.numstats=0;
        this.vocab = vocab;         
    }
    this.initialState = function (K, nw, nwsum) {
        var i;
        var M = this.documents.length;
        this.nw = nw; 
        this.nwsum = nwsum; 
    }
    
    this.predict_next_word = function (K,alpha,beta, nw, nwsum) {
      var i;
      this.K = K;
      this.alpha = alpha;
      this.beta = beta;
      
      this.initialState(K, nw, nwsum);
      
      var logNumParticles = Math.log(this.ITERATIONS);
      var totalLogLikelihood = 0;
      var tokenCount = 0;

      var m = 0;
      console.log('Processing doc #'+m);
      tokenCount += this.documents[m].length;
      
      var str = this.documents[m].reduce((acc, val)=>{
        return acc + ' ' + this.vocab[val];
      }, '');
      console.log(str);

      var sampling = this.left_to_right_sampling(m);
      return sampling;
    };

    this.left_to_right_sampling = function(m) {
      var docLength = this.documents[m].length;
      var topicAssignments = makeArray(docLength);
      
      // Create copy of stat counters for each iteration to a document
      // so it gets cleanup after evaluate each document.
      var nw = make2DArray(this.V,this.K); 
      var nd = make2DArray(this.documents.length,this.K); 
      var nwsum = makeArray(this.K); 
      var ndsum = makeArray(this.documents.length);
      var z = makeArray(docLength);
      for(var i=0;i<docLength;i++) {
        z[i] = null;
      } // for i
      for (var k = 0; k < this.K; k++) {
        nwsum[k] = this.nwsum[k];
        for (var v = 0; v < this.V; v++) {
          nw[v][k] = this.nw[v][k];
        }
      }


      var limit = docLength;
        
      for(var position=0;position<limit;position++) {          
        // Disregard words those are not in dictionary.
        //console.log('w = '+this.documents[m][position]);
        if(this.documents[m][position] == -1)
          continue;
        //console.log('re-sample on position: '+position);
        var sampling = this.sampleFullConditional(m, position, true,
          nw,nd,nwsum,ndsum,z
        );      
        topicAssignments[position] = sampling.topic;         
      }

      var sampling = this.predictNextWord(m,
        nw,nd,nwsum,ndsum,z
      );
      sampling.topics = topicAssignments;

      // console.log(' : Sampling:');
      var str = this.documents[m].reduce((acc, val, index)=>{
        if(index <= limit)
          return acc + ' ' + this.vocab[val] + '('+topicAssignments[index]+')';
        else
          return acc;
      }, '');
      console.log(' : ' + str + ' ['+this.vocab[sampling.word]+'][PROB = ' + (sampling.prob[sampling.word]*100).toFixed(2) +'%]');
      return sampling;
    };
    
    this.sampleFullConditional = function(m,n, is_resampling,
        nw,nd,nwsum,ndsum,z
      ) {
        var wordProbabilities = 0;
        var topic = 0;
        /*
        if(is_resampling) {
          topic = z[n];
          nw[this.documents[m][n]][topic]--;
          nd[m][topic]--;
          nwsum[topic]--;
          ndsum[m]--;
        }
        */
        var p = makeArray(this.K);
        for (var k = 0; k < this.K; k++) {
          if(is_resampling) {
            //console.log('nwsum['+k+']='+nwsum[k]+', this.V='+this.v+', this.beta='+this.beta);
            p[k] = (nw[this.documents[m][n]][k] + this.beta) / (nwsum[k] + this.V * this.beta)
              * (nd[m][k] + this.alpha) / (ndsum[m] + this.K * this.alpha);
          } else {
            p[k] = (nw[this.documents[m][n]][k] + this.beta) / (nwsum[k] + this.V * this.beta)
              * (nd[m][k] + this.alpha) / (ndsum[m] + this.K * this.alpha);
          }
          //if(!is_resampling) {
          // console.log('     p[topic='+k+'] = '+(p[k]*100).toFixed(2)+'%');
          //}
          wordProbabilities += p[k];
        }

        //if(!is_resampling)
        //  console.log('         p_sum['+0+'] = '+p[0]);
        for (var k = 1; k < p.length; k++) {
            p[k] += p[k - 1];    
            //if(!is_resampling) 
            //  console.log('         p_sum['+k+'] = '+p[k]);        
        }
        var u = this.getRandom() * p[this.K - 1];
        for (topic = 0; topic < p.length; topic++) {
          if (u < p[topic])
              break;
        }
        //if(!is_resampling) 
        //  console.log('           rand = '+u+', topic = '+topic);                
        z[n] = topic;
        //console.log('Got sample z['+n+'] = '+topic);
        nw[this.documents[m][n]][topic]++;
        nd[m][topic]++;
        nwsum[topic]++;
        ndsum[m]++;
        return {topic: topic, prob: wordProbabilities};
    }

    this.predictNextWord = function(m,
      nw,nd,nwsum,ndsum,z
      ) {
        var word = 0;
        var p = makeArray(this.V);
        for (var k = 0; k < this.K; k++) {
          for (var w = 0; w < this.V; w++) {
            p[w] += (nw[w][k] + this.beta) / (nwsum[k] + this.V * this.beta)
              * (nd[m][k] + this.alpha) / (ndsum[m] + this.K * this.alpha);
          }
        }

        //if(!is_resampling)
        //  console.log('         p_sum['+0+'] = '+p[0]);
        var sorted_p = [];        
        for (var k = 0; k < p.length; k++) {
          sorted_p.push({word:k, prob:p[k]})
        }
        sorted_p.sort(function(a, b) {
          if (a.prob < b.prob)
            return 1;
          if (a.prob > b.prob)
            return -1;
          return 0;          
        });
        console.log(JSON.stringify(sorted_p));
        return {word: sorted_p[0].word, prob: p, sort_p: sorted_p};
    }
  
    this.getRandom = function() {
        if (this.RANDOM_SEED) {
            // generate a pseudo-random number using a seed to ensure reproducable results.
            var x = Math.sin(this.RANDOM_SEED++) * 1000000;
            return x - Math.floor(x);
        } else {
            // use standard random algorithm.
            return Math.random();
        }
    }
}

module.exports = process;
