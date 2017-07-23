var express = require('express');
var router = express.Router();

router.get('/',function(req,res){
    res.render("main");
    //res.send("hello world");
})

module.exports = router;