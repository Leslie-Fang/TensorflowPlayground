var gulp = require('gulp');
var uglify = require('gulp-uglify');
var exec = require('child_process').exec;
var server = require( 'gulp-develop-server');
var babel = require('gulp-babel');

var Paths = {
    routes_src:'routes/*.js',
    routes_dest:'build/routes',
    html_src:'views/**',
    react_src:'public/javascript/react/**/*.js',
    react_dest: 'public/javascript/babel',
    gulp_src:'./gulpfile.js'

};

gulp.task('routes',function(){
    gulp.src(Paths.routes_src)
        .pipe(uglify())
        .pipe(gulp.dest(Paths.routes_dest));
});

gulp.task('babel',function(){
    gulp.src(Paths.react_src)
        .pipe(babel())
        .pipe(gulp.dest(Paths.react_dest));
});
gulp.task('webpack', function (cb) {
    exec('webpack', function (err, stdout, stderr) {
        console.log(stdout);
        console.log(stderr);
        cb(err);
    });
});
// run server
gulp.task( 'server:start', function() {
    server.listen( { path: './app.js' } );
});

// run server
gulp.task( 'server.restart', function() {
    server.restart();
});

gulp.task('watch',function(){
    gulp.watch([Paths.routes_src,Paths.html_src,Paths.react_src,Paths.gulp_src],['routes','babel','webpack','server.restart']);
});

gulp.task('default', ['routes','babel','webpack','server:start','watch']);