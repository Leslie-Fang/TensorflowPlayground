var gulp = require('gulp');
var uglify = require('gulp-uglify');
var exec = require('child_process').exec;
var server = require( 'gulp-develop-server');

var Paths = {
    routes_src:'routes/*.js',
    routes_dest:'build/routes',
    html_src:'views/**',
    js_src:'public/javascript/**'
};

gulp.task('routes',function(){
    gulp.src(Paths.routes_src)
        .pipe(uglify())
        .pipe(gulp.dest(Paths.routes_dest));
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
    gulp.watch([Paths.routes_src,Paths.html_src,Paths.js_src],['routes','server.restart']);
});

gulp.task('default', ['routes','server:start','watch']);