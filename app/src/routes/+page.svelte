<script>
    import { onMount } from 'svelte';
    import { tweened } from 'svelte/motion';
    import { cubicOut } from 'svelte/easing';
    import { tick } from 'svelte';
    import * as PIXI from 'pixi.js';
    import Comm from './Comm.svelte';
    let canvas;
    let ship_sprite;
    let ship_sprite_inner;
    let main_container;
    let word_log = [];
    let bottles = [
        {x: 600, y: 600, message: "hello"},
        {x: 800, y: 600, message: "hello"},
        {x: 900, y: 900, message: "world"}
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
    $: frames_x = Math.floor($tweened_ship_pos.x / FRAME_SIZE);
    $: frames_y = Math.floor($tweened_ship_pos.y / FRAME_SIZE);
    $: tweened_pan.set({x: frames_x * FRAME_SIZE, y: frames_y * FRAME_SIZE});

    let score = 0;

    let topics = [
        [{left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}, {left: "happy", right: "sad", top: "tame", bottom: "chaotic", category: "politics"}],
        [{left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}, {left: "evil", right: "good", top: "tame", bottom: "chaotic", category: "politics"}]
    ]
    $: current_topic = topics[frames_y][frames_x];
    let words_to_vecs;

    // let dummy_island_names_list = [dummy_island_names_1, dummy_island_names_2];
    let pickup_dist = 100;
    let islands = [];
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
    // let tweened_bottle_alpha = tweened(1, {duration: 1000, easing: cubicOut});
    // $: if (mounted && bottle_sprites.length > 0) bottle_sprites[current_bottle_index].alpha = $tweened_bottle_alpha;
    // $: dist = (mounted && bottle_sprites.length > 0) ? Math.sqrt((bottle_sprites[current_bottle_index].x - ship_sprite.x) ** 2 + (bottle_sprites[current_bottle_index].y - ship_sprite.y) ** 2) : 1024;
    // $: if (mounted && dist < pickup_dist && bottle_sprites[current_bottle_index].visible) {
    //     console.log("Picked up bottle");
    //     hide_bottle(); // can't do it inline to avoid circular dependency warning
    //     // bottle_sprite.visible = false;
    // }
    // function hide_bottle() {
    //     // bottle_sprite.visible = false;
    //     let done_tweening = tweened_bottle_alpha.set(0.2, {duration: 1000});
    //     done_tweening.then(() => {
    //         current_bottle_index += 1;
    //         bottle_sprites[current_bottle_index].visible = false;
    //         tweened_bottle_alpha.set(1, {duration: 0});
    //     });
    //     score += 1;
    // }
    function check_legal_move() {
        // don't let the ship go onto land
        let current_pos = {x: ship_sprite.x, y: ship_sprite.y};
        let new_pos = {x: current_pos.x + Math.cos(ship_sprite.rotation + Math.PI / 2) * 10, y: current_pos.y + Math.sin(ship_sprite.rotation + Math.PI / 2) * 10};
        // check if the new position is on any island
        for (let island of islands) {
            // check if the new position is in the island bounding box
            if (island.getBounds().containsPoint(new_pos.x, new_pos.y)) {
                // check each individual land tile
                let land_children_container = island.children[0];
                for (let child of land_children_container.children) {
                    if (child.getBounds().containsPoint(new_pos.x, new_pos.y)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    onMount(async () => {
        let screen_data = await fetch("/screen1.json");
        screen_data = await screen_data.json();
        console.log(screen_data);

        words_to_vecs = await fetch("/s1.json");
        words_to_vecs = await words_to_vecs.json();
        console.log("revolution", words_to_vecs["revolution"]);

        const app = new PIXI.Application();
        await app.init({ canvas: canvas, width: 1024, height: 1024, backgroundColor: 0x1099bb });

        await PIXI.Assets.init({ manifest: "/manifest.json"}); // need to load manifest before we can reference bundles
        let tiles = await PIXI.Assets.loadBundle("tiles");
        console.log(tiles);

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
        let tiling_sprite = new PIXI.TilingSprite({ texture: tilesheet, width: 2048, height: 2048 });
        main_container.addChild(tiling_sprite);

        // Create a new Graphics object
        let graphics = new PIXI.Graphics();
        // graphics.moveTo(-200, 600);
        graphics.moveTo(...screen_data.track_bezier[0].start);
        // graphics.bezierCurveTo(100, 100, 200, 200, 1224, 600);
        for (let curve of screen_data.track_bezier) {
            graphics.bezierCurveTo(...curve.cp1, ...curve.cp2, ...curve.end);
        }
        graphics.stroke({width: 140, color: 0xaaaaff, alpha: 0.4});
        main_container.addChild(graphics);
        //
        for (let island_data of screen_data.islands) {
            console.log("Creating island", island_data);
            let island_container = new PIXI.Container();
            let i1land = new PIXI.Container();
            let i1water = new PIXI.Container();
            island_container.addChild(i1land);
            island_container.addChild(i1water);
            for (let i = 0; i < island_data.tiles.length; i++) {
                for (let j = 0; j < island_data.tiles[i].length; j++) {
                    let island_sprite = PIXI.Sprite.from(tiles[island_data.tiles[i][j]]);
                    island_sprite.x = j * 128;
                    island_sprite.y = i * 128;
                    if (island_data.tiles[i][j] === "water") {
                        // i1water.addChild(island_sprite);
                    } else {
                        i1land.addChild(island_sprite);
                    }
                }
            }
            // add text to the container
            let text = new PIXI.Text({text: "isl", fontFamily : 'serif', fontSize: 100, fill : 0xff1010, align : 'center'});
            text.x = island_container.width / 2;
            text.y = island_container.height / 2;
            text.anchor.set(0.5);
            island_container.addChild(text);
            // pos
            island_container.position = island_data.position;
            island_container.pivot.set(island_container.width / 2, island_container.height / 2);
            island_container.rotation = island_data.rotation;
            // island_container.rotation = 3.14 / 6.0;
            // scale down 50%
            island_container.scale.set(0.5);
            islands.push(island_container);
            main_container.addChild(island_container);
        }

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


        // test svg
        const test_svg = new PIXI.Graphics().svg(`
            <svg height="1040px" width="2940px" xmlns="http://www.w3.org/2000/svg" >
                <path id="curve1" stroke="blue" fill="none" stroke-width="4" 
                d="M -266.05404,678.05272 C 230.77585,63.840764 1225.4074,603.54864 1095.1485,1330.6781 c 324.1766,517.9907 1066.4264,422.3671 1565.9771,268.9905"
                />
            </svg>
        `);
        main_container.addChild(test_svg);

        // TODO
        add_arrow_to_ship("hello", {x: 100, y: 100});
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
            if (check_legal_move()) {
                // ship_sprite.x += Math.cos(ship_sprite.rotation + Math.PI / 2) * 10;
                // ship_sprite.y += Math.sin(ship_sprite.rotation + Math.PI / 2) * 10;
                let rate = 160.0;
                tweened_ship_pos.set({x: ship_sprite.x + Math.cos(ship_sprite_inner.rotation + Math.PI / 2) * rate, y: ship_sprite.y + Math.sin(ship_sprite_inner.rotation + Math.PI / 2) * rate});
            }
            event.preventDefault();
        }
    }
    function handleEnter(event) {
        if (event.key === "Enter") {
            console.log("Enter pressed", event.target.value, words_to_vecs[event.target.value]);
            let delta = words_to_vecs[event.target.value];
            let factor = 20.0;
            // ship_sprite.x += delta[0] * factor;
            tweened_ship_pos.set({x: ship_sprite.x + delta[0] * factor, y: ship_sprite.y + delta[1] * factor});
            tweened_pan.set({x: 100, y: 100});
            // ship_sprite.y += delta[1] * factor;
            event.preventDefault();
        }
    }
    function handleReceiveMessage(event) {
        let word = event.detail.message;
        word_log = [...word_log, word];
        console.log("Received message", word);
        let delta = words_to_vecs[word];
        let factor = 20.0;
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
    }

</script>

<div class="hidden">
    <Comm messages={["m1", "m2", "m3"]} on:receive_message={handleReceiveMessage} />
</div>

<div class="game h-screen w-screen bg-gray-500 flex flex-row items-center justify-center relative" >
    <div class="h-[1024px] flex flex-row gap-8">
        <canvas class="h-full w-full border-4 border-black" id="app" bind:this={canvas}></canvas>
        <div class="h-full w-[400px] bg-green-300 border-4 border-black text-3xl flex flex-col justify-between gap-0 p-0">
            <input type="text" class="w-full p-0 m-0" placeholder="Type here" on:keydown={handleEnter}
            />
            {JSON.stringify(frames_x)} {JSON.stringify(frames_y)}
            {JSON.stringify(current_topic)}
            {JSON.stringify(word_log)}
            <div class="word_log bg-gray-200" >
                {#each word_log as word}
                    <div class="word">{word}</div>
                {/each}
            </div>
            <br>
            <br>
            <div class="bottle_messages w-full">
                {#each bottles.slice(0, current_bottle_index) as bottle}
                    <div class="bottle_message">
                        {bottle.message}
                    </div>
                {/each}
            </div>
            <!-- compass rose -->
            <div class="rose_container w-full flex flex-row justify-center mb-16">
                <div class="compass_rose w-64 h-64 bg-contain bg-no-repeat square bg-[url('/rose.png')] relative ">
                    <div class="north absolute -top-4 left-1/2 transform -translate-y-1/2 -translate-x-1/2              font-serif">{current_topic.top}</div>
                    <div class="east absolute top-1/2 -right-4 transform translate-x-1/2 -translate-y-1/2 rotate-90     font-serif">{current_topic.right}</div>
                    <div class="south absolute -bottom-4 left-1/2 transform translate-y-1/2 -translate-x-1/2 rotate-180 font-serif">{current_topic.bottom}</div>
                    <div class="west absolute bottom-1/2 -left-4 transform -translate-x-1/2 translate-y-1/2 -rotate-90  font-serif">{current_topic.left}</div>
                </div>
            </div>
        </div>
    </div>
</div>
