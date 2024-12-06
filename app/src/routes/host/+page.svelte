<script>
    import { onMount } from 'svelte';
    import { tweened } from 'svelte/motion';
    import { cubicOut } from 'svelte/easing';
    import { tick } from 'svelte';
    import * as PIXI from 'pixi.js';
    import Comm from '../Comm.svelte';
    let svg_file_name = "/drawing.svg";
    let background_png_file_name = "/drawing.png";
    let canvas;
    let mySVG;
    let svg;
    let svg_paths;
    let ship_sprite;
    let ship_sprite_inner;
    let main_container;
    let word_log = [];
    let bottles = [
        {x: 900, y: 850, message: "You"},
        {x: 1900, y: 550, message: "win"},
        {x: 2500, y: 600, message: "the game!"},
        {x: 2400 + 600, y: 1200 + 600, message: "secret"},
        {x: 1024 + 600, y: 1024 + 600, message: "message!"},
        {x: 5000, y: 5000, message: "neverland"}
    ];
    $: current_bottle_index = 0;
    $: current_bottle = bottles[current_bottle_index];
    let bottle_sprites = [];
    let FRAME_SIZE = 1024;
    let pan = {x: 0, y: 0};
    let tweened_pan = tweened(pan, {duration: 1000, easing: cubicOut});
    $: if (main_container) main_container.position = {x: -$tweened_pan.x, y: -$tweened_pan.y};
    let ship_pos = {x: 1024.0 / 2.0, y: 1024.0 / 2.0};
    let tweened_ship_pos = tweened(ship_pos, {duration: 1000, easing: cubicOut});
    $: if (ship_sprite) ship_sprite.position = $tweened_ship_pos;
    // make the tweened_pan react to the tweened ship pos
    $: frames_x = Math.min(2, Math.max(0, Math.floor($tweened_ship_pos.x / FRAME_SIZE)));
    $: frames_y = Math.min(2, Math.max(0, Math.floor($tweened_ship_pos.y / FRAME_SIZE)));
    $: tweened_pan.set({x: frames_x * FRAME_SIZE, y: frames_y * FRAME_SIZE});
    $: console.log(frames_x, frames_y);

    let score = 0;

    let topics = [
        [{left: "aquatic", right: "terrestrial", top: "huge", bottom: "tiny", category: "animals"},
         {left: "intense", right: "mild", top: "positive", bottom: "negative", category: "emotions"},
         {left: "evil", right: "good", top: "chaos", bottom: "order", category: "politics"}
        ],
        [ {left: "happy", right: "sad", top: "tame", bottom: "chaotic", category: "c1"},
         {left: "cold", right: "hot", top: "wet", bottom: "dry", category: "Weather"},
         {left: "ancient", right: "modern", top: "art", bottom: "science", category: "Academic Disciplines"},
        ],
        [{left: "evil", right: "good", top: "order", bottom: "chaos", category: "c3"},
         {left: "happy", right: "sad", top: "tame", bottom: "chaotic", category: "d3"},
         {left: "happy", right: "sad", top: "tame", bottom: "chaotic", category: "e3"}
        ]
    ]
    $: current_topic = topics[frames_y][frames_x];
    // $: current_topic = topics[1][1];
    $: console.log(current_topic);
    let dicts_array = [
        [{}, {}, {}],
        [{}, {}, {}],
        [{}, {}, {}]
    ]
    $: words_to_vecs = dicts_array[frames_y][frames_x];

    let pickup_dist = 100;
    let mounted = false;
    // DISTANCE PICKUP
    $: update_bottle_pickup($tweened_ship_pos);
    function update_bottle_pickup(ship_pos) {
        if (!bottle_sprites.length) {
            return;
        }
        let current_bottle = bottle_sprites[current_bottle_index];
        let dist = Math.sqrt((current_bottle.x - ship_pos.x) ** 2 + (current_bottle.y - ship_pos.y) ** 2);
        if (dist < pickup_dist) {
            console.log("Picked up bottle");
            current_bottle.alpha = 0.2;
            current_bottle_index += 1;
        }
    }
    onMount(async () => {
        // let screen_data = await fetch("/screen1.json");
        // screen_data = await screen_data.json();
        // console.log(screen_data);

        let g1 = await fetch("/animals_coordinates.json");
        dicts_array[0][0] = await g1.json();
        let g2 = await fetch("/emotions_coordinates.json");
        dicts_array[0][1] = await g2.json();
        let g3 = await fetch("/politics_coordinates.json");
        dicts_array[0][2] = await g3.json();
        let g4 = await fetch("/g4.json");
        dicts_array[1][2] = await g4.json();
        let g5 = await fetch("/g5.json");
        dicts_array[1][1] = await g5.json();

        const app = new PIXI.Application();
        await app.init({ canvas: canvas, width: 1024, height: 1024, backgroundColor: 0x1099bb });

        await PIXI.Assets.init({ manifest: "/manifest.json"}); // need to load manifest before we can reference bundles
        let tiles = await PIXI.Assets.loadBundle("tiles");

        main_container = new PIXI.Container();
        // main_container.scale.set(0.5); // TODO ZOOM IN ON INTRO
        app.stage.addChild(main_container);

        const tilesheet  = await PIXI.Assets.load('/kenney_pirate/Tilesheet/tiles_sheet.png');
        let frame = tilesheet.frame;
        frame.width = 64;
        frame.height = 64;
        frame.x = 64 * 8;
        frame.y = 64 * 4;
        tilesheet.updateUvs();
        let tiling_sprite = new PIXI.TilingSprite({ texture: tilesheet, width: 3072, height: 3072 });
        main_container.addChild(tiling_sprite);

        let svg_file = await fetch(svg_file_name);
        let svg_text = await svg_file.text();
        // test svg
            // <svg height="1040px" width="2940px" xmlns="http://www.w3.org/2000/svg" >
            //     <path id="curve1" stroke="blue" fill="none" stroke-width="4" 
            //     d="M -266.05404,678.05272 C 230.77585,63.840764 1225.4074,603.54864 1095.1485,1330.6781 c 324.1766,517.9907 1066.4264,422.3671 1565.9771,268.9905"
            //     />
            // </svg>
        let graphics = new PIXI.Graphics();
        const test_svg = graphics.svg( svg_text);
        main_container.addChild(test_svg);

        // background png
        const background_texture = await PIXI.Assets.load(background_png_file_name);
        const background_sprite = PIXI.Sprite.from(background_texture);
        main_container.addChild(background_sprite);



        // Create a new Graphics object
        // let graphics = new PIXI.Graphics();
        // graphics.moveTo(-200, 600);
        // graphics.moveTo(...screen_data.track_bezier[0].start);
        // graphics.bezierCurveTo(100, 100, 200, 200, 1224, 600);
        // for (let curve of screen_data.track_bezier) {
            // graphics.bezierCurveTo(...curve.cp1, ...curve.cp2, ...curve.end);
        // }
        // graphics.stroke({width: 140, color: 0xaaaaff, alpha: 0.4});
        // main_container.addChild(graphics);
        //
        for (let bottle of bottles) {
            let bottle_sprite = PIXI.Sprite.from(tiles.bottle);
            bottle_sprite.x = bottle.x;
            bottle_sprite.y = bottle.y;
            bottle_sprite.anchor.set(0.5);
            bottle_sprites.push(bottle_sprite);
            main_container.addChild(bottle_sprite);
        }
        

        // ship sprite
        await PIXI.Assets.load('/kenney_pirate/PNG/Retina/Ships/ship (1).png');
        // ship_sprite = PIXI.Sprite.from('/kenney_pirate/PNG/Retina/Ships/ship (1).png');
        ship_sprite = new PIXI.Container();
        ship_sprite_inner = PIXI.Sprite.from('/kenney_pirate/PNG/Retina/Ships/ship (1).png');
        ship_sprite_inner.anchor.set(0.5);
        ship_sprite.addChild(ship_sprite_inner);
        ship_sprite.x = 1024 / 2;
        ship_sprite.y = 1024 / 2;
        main_container.addChild(ship_sprite);

        // svg for collision testing
        svg = mySVG.contentDocument.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg_paths = mySVG.contentDocument.querySelectorAll('path');

        // app.stage.addChild(sprite);
        app.ticker.add((ticker) => {
            // tilesheet_sprite.rotation +=  0.01 * ticker.deltaTime;
        });

        window.addEventListener('keydown', handleKeyPress);
        mounted = true;
        return () => {
            app.destroy();
            window.removeEventListener('keydown', handleKeyPress); 
        };
    });
    function collides_land(pos) {
        // TODO: a real barrier instead of a sandbar
        if (!svg_paths) {
            return false;
        }
        const point_obj = svg.createSVGPoint();
        point_obj.x = pos.x;
        point_obj.y = pos.y;
        let first = true;
        for (let path of svg_paths) {
            if (first) {
                // skip the first path which is the water route
                first = false;
                continue;
            }
            let is_inside = path.isPointInFill(point_obj);
            if (is_inside) {
                // arrest motion
                tweened_ship_pos.set($tweened_ship_pos, {duration: 0});
                return true;
            }
        }
        return false;
    }
    $: console.log(collides_land($tweened_ship_pos));
    function handleKeyPress(event) {
        // right
        let key_to_rotation = {
            ArrowDown: 0,
            ArrowUp: Math.PI,
            ArrowLeft: Math.PI / 2,
            ArrowRight: -Math.PI / 2
        };
        if (event.key in key_to_rotation) {
            ship_sprite_inner.rotation = key_to_rotation[event.key];
            let rate = 160.0;
            tweened_ship_pos.set({x: ship_sprite.x + Math.cos(ship_sprite_inner.rotation + Math.PI / 2) * rate, y: ship_sprite.y + Math.sin(ship_sprite_inner.rotation + Math.PI / 2) * rate});
            event.preventDefault();
        }
    }
    // function handleEnter(event) {
    //     if (event.key === "Enter") {
    //         console.log("Enter pressed", event.target.value, words_to_vecs[event.target.value]);
    //         let delta = words_to_vecs[event.target.value];
    //         let factor = 20.0;
    //         // ship_sprite.x += delta[0] * factor;
    //         tweened_ship_pos.set({x: ship_sprite.x + delta[0] * factor, y: ship_sprite.y + delta[1] * factor});
    //         tweened_pan.set({x: 100, y: 100});
    //         // ship_sprite.y += delta[1] * factor;
    //         event.preventDefault();
    //     }
    // }
    function handleReceiveMessage(event) {
        let word = event.detail.message;
        word = word.toLowerCase();
        if (word_log.includes(word)) {
            // guard word already logged
            return;
        }
        word_log = [...word_log, word];
        console.log("Received message", word);
        let delta = words_to_vecs[word];
        if (!delta) {
            // guard word in dictionary
            return;
        }
        let factor = 920.0;
        tweened_ship_pos.set({x: ship_sprite.x + delta[0] * factor, y: ship_sprite.y + delta[1] * factor});
        //
        add_arrow_to_ship(word, {x: delta[0] * factor, y: delta[1] * factor});
        // ship_sprite.x += delta[0] * factor;
        // ship_sprite.y += delta[1] * factor;
    }
    function add_arrow_to_ship(text, dp) {
        // add an arrow to the ship
        let arrow_container = new PIXI.Container();
        let word_text = new PIXI.Text(text, {fontFamily: 'serif', fontSize: 60, fill : 0xff1010, align : 'center'});
        word_text.x = 50;
        word_text.y = -40;
        arrow_container.pivot.set(0, 0);
        arrow_container.rotation = Math.atan2(dp.y, dp.x);
        arrow_container.addChild(word_text);
        ship_sprite.addChild(arrow_container);
        // hmmm, we want to use absolute rotation for the arrow container which means it needs to be a sibling not a child of the ship
        let fade = setInterval(() => {
            arrow_container.alpha *= 0.95;
        }, 50);
        setTimeout(() => {
            clearInterval(fade);
            ship_sprite.removeChild(arrow_container);
        }, 3000);
    }
    function capitalize(word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
    }

</script>

<div class="hidden">
    <Comm messages={["m1", "m2", "m3"]} on:receive_message={handleReceiveMessage} />
    <object data={svg_file_name} type="image/svg+xml" bind:this={mySVG}></object>
</div>

<div class="game h-screen w-screen bg-gray-500 flex flex-row items-center justify-center relative" >
    <div class="h-[1024px] flex flex-row gap-8">
        <canvas class="h-full w-full border-4 border-black" id="app" bind:this={canvas}></canvas>
        <div class="h-full w-[400px] bg-green-600 border-4 border-black text-3xl flex flex-col justify-between gap-0 p-0">
            <!--
            <input type="text" class="w-full p-0 m-0" placeholder="Type here" on:keydown={handleEnter}
            />
            {JSON.stringify(frames_x)} {JSON.stringify(frames_y)}
            {JSON.stringify(current_topic)}
            {JSON.stringify(word_log)}
            -->
            <div class="word_log bg-gray-200 flex-1 flex flex-col bg-white m-4 rounded-lg max-h-1/2 overflow-y-scroll p-4" >
                <div class="text-4xl font-serif">
                    Ship Log
                </div>
                <hr>
                {#each [...word_log].reverse() as word}
                    <div class="word font-serif">{word}</div>
                {/each}
            </div>
            <div class="word_log bg-gray-200 flex-1 flex flex-col bg-white m-4 rounded-lg max-h-1/2 overflow-y-scroll p-4" >
                <div class="text-4xl font-serif">
                    Message in a Bottle
                </div>
                <hr>
                {#each bottles.slice(0, current_bottle_index) as bottle}
                    <div class="bottle_message">
                        {bottle.message}
                    </div>
                {/each}
            </div>
            <!-- compass rose -->
            <div class="rose_container flex flex-col items-center justify-center m-4 rounded-lg bg-white p-4">
                <div class="text-4xl font-serif">
                    {capitalize(current_topic.category)}
                </div>
                <hr>
                <div class="compass_rose w-64 h-64 bg-contain bg-no-repeat square bg-[url('/rose.png')] relative m-16">
                    <!-- <div class="north absolute -top-16 left-1/2 transform -translate-y-1/2 -translate-x-1/2              font-serif text-4xl">{capitalize(current_topic.category)}</div> -->
                    <div class="north absolute -top-4 left-1/2 transform -translate-y-1/2 -translate-x-1/2              font-serif">{capitalize(current_topic.top)}</div>
                    <div class="east absolute top-1/2 -right-4 transform translate-x-1/2 -translate-y-1/2 rotate-90     font-serif">{capitalize(current_topic.right)}</div>
                    <div class="south absolute -bottom-4 left-1/2 transform translate-y-1/2 -translate-x-1/2 rotate-180 font-serif">{capitalize(current_topic.bottom)}</div>
                    <div class="west absolute bottom-1/2 -left-4 transform -translate-x-1/2 translate-y-1/2 -rotate-90  font-serif">{capitalize(current_topic.left)}</div>
                </div>
            </div>
        </div>
    </div>
</div>
